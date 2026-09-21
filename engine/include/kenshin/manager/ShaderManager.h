#pragma once

// ── Shader manager (Slang -> SPIR-V runtime compilation, S1.4) ──
// Wraps the vendored Slang runtime library. Raw Slang types stay in the .cpp
// (same discipline as the backend: no third-party API types leak here).

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace manager {

/// One reflected descriptor binding (plain data; the pipeline layer maps
/// these to backend descriptor-set layouts instead of hand-maintaining a
/// second copy of what the shader already declares).
struct ReflectedBinding {
  u32 set = 0;
  u32 binding = 0;
  enum class Type {
    StorageBuffer,
    StorageImage,
    CombinedImageSampler,  // Slang Sampler2D (combined)
    SampledTexture,        // SPIR-V SAMPLED_IMAGE (separate sampler)
    Sampler,
    UniformBuffer,
    Unknown,
  };
  Type type = Type::Unknown;
};

/// Reflect descriptor bindings from compiled SPIR-V (spirv-reflect).
/// Returns false on parse failure. An empty result is valid (e.g.
/// push-constant-only shaders like the baseline pass).
bool reflectDescriptorBindings(const std::vector<u32>& spirv,
                               std::vector<ReflectedBinding>& out);

class ShaderManager {
 public:
  /// Create the Slang global session and a session targeting SPIR-V.
  /// @param shaderRoot  directory used to resolve #include and relative
  ///                    file loads (the engine's shaders/ directory).
  bool init(const std::string& shaderRoot);
  void shutdown();
  /// Compile a .slang file (absolute path, or relative to shaderRoot)
  /// for the entry point `entryPoint` ("main" by default).
  /// On failure fills `log` with the compiler diagnostics and returns false.
  bool compileFile(const std::string& path, std::vector<u32>& spirvOut,
                   std::string& log, const char* entryPoint = "main");

  /// Compile inline source (error-path tests, snippets).
  bool compileSource(const std::string& source, const std::string& name,
                     std::vector<u32>& spirvOut, std::string& log);

  const std::string& shaderRoot() const { return m_shaderRoot; }

 private:
  /// Compile-cache key: the resolved shader file, the entry point and the
  /// file's mtime (a future hot reload just mtime-bumps its way in).
  struct SpirvCacheKey {
    std::string file;
    std::string entryPoint;
    i64 mtime = 0;

    bool operator<(const SpirvCacheKey& o) const {
      return std::tie(file, entryPoint, mtime) <
             std::tie(o.file, o.entryPoint, o.mtime);
    }
  };

  void* m_globalSession = nullptr;  // ISlangGlobalSession*
  void* m_session = nullptr;        // ISlangSession*
  /// One slang::IModule* per successful loadModule call, kept alive for
  /// the session's lifetime (the session's name cache does not hold its
  /// own reference - see compileFile) and released in shutdown().
  std::vector<void*> m_liveModules;
  /// Compiled SPIR-V per (file, entry point, mtime): the one-shot chain
  /// re-compiles its shaders on every view change - serve repeats from
  /// here instead of re-linking. Single-threaded app, no locking.
  std::map<SpirvCacheKey, std::vector<u32>> m_spirvCache;
  std::string m_shaderRoot;
};

}  // namespace manager

KENSHIN_NS_END
