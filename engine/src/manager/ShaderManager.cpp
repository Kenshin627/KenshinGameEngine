#include "kenshin/manager/ShaderManager.h"

#include "kenshin/core/Log.h"

#pragma warning(push, 0)
#include <slang.h>
#pragma warning(pop)

#include <cstdio>
#include <filesystem>

KENSHIN_NS_BEGIN

namespace manager
{

    namespace
    {
        std::string blobText(slang::IBlob* blob)
        {
            if (!blob || blob->getBufferSize() == 0)
                return {};
            return std::string(static_cast<const char*>(blob->getBufferPointer()),
                               static_cast<usize>(blob->getBufferSize()));
        }
    } // namespace

    bool ShaderManager::init(const std::string& shaderRoot)
    {
        m_shaderRoot = shaderRoot;

        slang::IGlobalSession* globalSession = nullptr;
        if (const SlangResult result = slang_createGlobalSession(SLANG_API_VERSION, &globalSession);
            result != SLANG_OK || !globalSession)
        {
            KENSHIN_ERROR("[slang] slang_createGlobalSession failed (0x{:X})",
                          static_cast<unsigned>(result));
            return false;
        }

        slang::SessionDesc desc = {};
        const char* searchPaths[1] = { shaderRoot.c_str() };
        desc.searchPathCount = 1;
        desc.searchPaths = searchPaths;

        slang::TargetDesc target = {};
        target.format = SLANG_SPIRV;
        target.profile = globalSession->findProfile("spirv_1_5");
        target.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
        desc.targetCount = 1;
        desc.targets = &target;

        slang::ISession* session = nullptr;
        if (const SlangResult result = globalSession->createSession(desc, &session);
            result != SLANG_OK || !session)
        {
            KENSHIN_ERROR("[slang] createSession failed (0x{:X})",
                          static_cast<unsigned>(result));
            globalSession->release();
            return false;
        }

        m_globalSession = globalSession;
        m_session = session;

        KENSHIN_INFO("[slang] session ready (SPIR-V 1.5 target, Slang 2026.3.1 vendored)");
        return true;
    }

    void ShaderManager::shutdown()
    {
        if (auto* session = static_cast<slang::ISession*>(m_session))
        {
            session->release();
            m_session = nullptr;
        }
        if (auto* globalSession = static_cast<slang::IGlobalSession*>(m_globalSession))
        {
            globalSession->release();
            m_globalSession = nullptr;
        }
    }

    bool ShaderManager::compileFile(const std::string& path, std::vector<u32>& spirvOut,
                                    std::string& log)
    {
        auto* session = static_cast<slang::ISession*>(m_session);
        if (!session)
        {
            log = "ShaderManager not initialized";
            return false;
        }

        // ISession resolves modules by NAME through its search paths (the
        // shader root); derive the module name from the file name.
        const std::string moduleName =
            std::filesystem::path(path).replace_extension().filename().string();
        if (!std::filesystem::exists(path))
        {
            std::error_code ec;
            const auto candidate = std::filesystem::path(m_shaderRoot) / path;
            if (!std::filesystem::exists(candidate, ec))
            {
                log = "shader file not found: " + path;
                return false;
            }
        }

        slang::IBlob* diagnostics = nullptr;
        slang::IModule* module = session->loadModule(moduleName.c_str(), &diagnostics);
        if (!module)
        {
            log = blobText(diagnostics);
            if (log.empty())
                log = "loadModule('" + moduleName + "') failed";
            if (diagnostics) diagnostics->release();
            return false;
        }

        slang::IEntryPoint* entryPoint = nullptr;
        if (const SlangResult result = module->findEntryPointByName("main", &entryPoint);
            result != SLANG_OK || !entryPoint)
        {
            log = blobText(diagnostics);
            if (log.empty())
                log = "entry point 'main' not found";
            if (diagnostics) diagnostics->release();
            module->release();
            return false;
        }

        slang::IComponentType* components[2] = { module, entryPoint };
        slang::IComponentType* program = nullptr;
        slang::IBlob* linkDiag = nullptr;
        if (const SlangResult result =
                session->createCompositeComponentType(components, 2, &program, &linkDiag);
            result != SLANG_OK || !program)
        {
            log = blobText(linkDiag);
            if (linkDiag) linkDiag->release();
            if (diagnostics) diagnostics->release();
            entryPoint->release();
            module->release();
            return false;
        }

        slang::IComponentType* linked = nullptr;
        slang::IBlob* linkDiag2 = nullptr;
        if (const SlangResult result = program->link(&linked, &linkDiag2);
            result != SLANG_OK || !linked)
        {
            log = blobText(linkDiag2);
            if (linkDiag2) linkDiag2->release();
            program->release();
            if (diagnostics) diagnostics->release();
            entryPoint->release();
            module->release();
            return false;
        }

        slang::IBlob* code = nullptr;
        slang::IBlob* codeDiag = nullptr;
        SlangResult result =
            linked->getEntryPointCode(0 /*entryPointIndex*/, 0 /*targetIndex*/,
                                      &code, &codeDiag);
        if (result == SLANG_OK && code && code->getBufferSize() > 0)
        {
            const usize byteSize = static_cast<usize>(code->getBufferSize());
            spirvOut.resize(byteSize / sizeof(u32));
            std::memcpy(spirvOut.data(), code->getBufferPointer(), byteSize);
        }
        else
        {
            log = blobText(codeDiag);
            if (log.empty())
                log = "getEntryPointCode failed";
            result = SLANG_FAIL;
        }

        if (codeDiag) codeDiag->release();
        if (code) code->release();
        linked->release();
        program->release();
        entryPoint->release();
        module->release();
        if (diagnostics) diagnostics->release();
        return result == SLANG_OK;
    }

    bool ShaderManager::compileSource(const std::string& source, const std::string& name,
                                      std::vector<u32>& spirvOut, std::string& log)
    {
        auto* session = static_cast<slang::ISession*>(m_session);
        if (!session)
        {
            log = "ShaderManager not initialized";
            return false;
        }

        slang::IBlob* diagnostics = nullptr;
        slang::IModule* module =
            session->loadModuleFromSourceString(name.c_str(), name.c_str(),
                                                source.c_str(), &diagnostics);
        if (!module)
        {
            log = blobText(diagnostics);
            if (diagnostics) diagnostics->release();
            return false;
        }

        slang::IEntryPoint* entryPoint = nullptr;
        if (module->findEntryPointByName("main", &entryPoint) != SLANG_OK || !entryPoint)
        {
            log = blobText(diagnostics);
            if (log.empty())
                log = "entry point 'main' not found";
            if (diagnostics) diagnostics->release();
            module->release();
            return false;
        }

        slang::IComponentType* components[2] = { module, entryPoint };
        slang::IComponentType* program = nullptr;
        session->createCompositeComponentType(components, 2, &program, nullptr);

        slang::IComponentType* linked = nullptr;
        SlangResult result = program ? program->link(&linked, nullptr) : SLANG_FAIL;
        if (result != SLANG_OK || !linked)
        {
            log = "link failed";
            if (program) program->release();
            entryPoint->release();
            module->release();
            if (diagnostics) diagnostics->release();
            return false;
        }

        slang::IBlob* code = nullptr;
        slang::IBlob* codeDiag = nullptr;
        result = linked->getEntryPointCode(0, 0, &code, &codeDiag);
        if (result == SLANG_OK && code && code->getBufferSize() > 0)
        {
            const usize byteSize = static_cast<usize>(code->getBufferSize());
            spirvOut.resize(byteSize / sizeof(u32));
            std::memcpy(spirvOut.data(), code->getBufferPointer(), byteSize);
        }
        else
        {
            log = blobText(codeDiag);
            result = SLANG_FAIL;
        }

        if (codeDiag) codeDiag->release();
        if (code) code->release();
        linked->release();
        program->release();
        entryPoint->release();
        module->release();
        if (diagnostics) diagnostics->release();
        return result == SLANG_OK;
    }

} // namespace manager

KENSHIN_NS_END
