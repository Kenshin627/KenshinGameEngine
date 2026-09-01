#include "kenshin/Engine.h"

#include <string>

KENSHIN_NS_BEGIN

Engine& Engine::get() {
  static Engine instance;
  return instance;
}

void Engine::beginFrame(f32 dt) {
  for (auto& [key, sub] : m_subsystems) sub->onBeginFrame(dt);
}

void Engine::endFrame() {
  for (auto& [key, sub] : m_subsystems) sub->onEndFrame();
}

void Engine::shutdown() {
  // Reverse registration order: a subsystem may only depend on ones
  // registered BEFORE it, so it must die BEFORE them.
  for (auto it = m_subsystems.rbegin(); it != m_subsystems.rend(); ++it)
    it->second->onShutdown();
  m_subsystems.clear();
}

void* Engine::adopt(std::type_index key, std::shared_ptr<ISubsystem> sub) {
  for (auto it = m_subsystems.begin(); it != m_subsystems.end(); ++it) {
    if (it->first == key) {
      it->second->onShutdown();
      m_subsystems.erase(it);
      break;
    }
  }
  void* impl = sub->implPtr();
  if (!sub->onInit(*this)) {
    throw std::runtime_error(std::string("Engine: subsystem '") + key.name() +
                             "' onInit failed");
  }
  m_subsystems.emplace_back(key, std::move(sub));
  return impl;
}

KENSHIN_NS_END
