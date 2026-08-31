#pragma once

// ── Shader manager (Slang -> SPIR-V runtime compilation, S1.4) ──
// Wraps the vendored Slang runtime library. Raw Slang types stay in the .cpp
// (same discipline as the backend: no third-party API types leak here).

#include "kenshin/Common.h"

#include <string>
#include <vector>

KENSHIN_NS_BEGIN

namespace manager
{

    class ShaderManager
    {
    public:
        /// Create the Slang global session and a session targeting SPIR-V.
        /// @param shaderRoot  directory used to resolve #include and relative
        ///                    file loads (the engine's shaders/ directory).
        bool init(const std::string& shaderRoot);
        void shutdown();

        /// Compile a .slang file (absolute path, or relative to shaderRoot).
        /// On failure fills `log` with the compiler diagnostics and returns false.
        bool compileFile(const std::string& path, std::vector<u32>& spirvOut,
                         std::string& log);

        /// Compile inline source (error-path tests, snippets).
        bool compileSource(const std::string& source, const std::string& name,
                           std::vector<u32>& spirvOut, std::string& log);

        const std::string& shaderRoot() const { return m_shaderRoot; }

    private:
        void* m_globalSession = nullptr; // ISlangGlobalSession*
        void* m_session = nullptr;       // ISlangSession*
        std::string m_shaderRoot;
    };

} // namespace manager

KENSHIN_NS_END
