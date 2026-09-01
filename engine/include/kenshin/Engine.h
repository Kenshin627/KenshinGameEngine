#pragma once

// ── Engine — facade singleton owning the subsystem registry ──────
// registerSubsystem<T>(args...) constructs T by value; registerInstance<T>
// adopts an existing unique_ptr (factory-built, interface-typed). get<T>()
// resolves by type from anywhere. Registration order is preserved and
// shutdown() releases subsystems in REVERSE order (construction symmetry),
// so later registrations may safely depend on earlier ones.

#include "kenshin/ISubsystem.h"

#include <memory>
#include <stdexcept>
#include <typeindex>
#include <utility>
#include <vector>

KENSHIN_NS_BEGIN

class Engine
{
public:
    /// Global access point; lives until shutdown().
    static Engine& get();

    /// Register a by-value subsystem; calls its optional onInit(Engine&)
    /// hook immediately. Throws std::runtime_error if onInit fails.
    template <typename T, typename... Args>
    T& registerSubsystem(Args&&... args)
    {
        auto sub = std::make_shared<Subsystem<T>>(std::forward<Args>(args)...);
        return *static_cast<T*>(adopt(std::type_index(typeid(T)), std::move(sub)));
    }

    /// Adopt a factory-built instance under key T (usually an interface).
    template <typename T>
    T& registerInstance(std::unique_ptr<T> impl)
    {
        auto sub = std::make_shared<PointerSubsystem<T>>(std::move(impl));
        return *static_cast<T*>(adopt(std::type_index(typeid(T)), std::move(sub)));
    }

    /// Resolve by type; throws when not registered.
    template <typename T>
    T& get()
    {
        if (T* p = find<T>())
            return *p;
        throw std::runtime_error(std::string("Engine::get(): subsystem '") +
                                 typeid(T).name() + "' not registered");
    }

    /// Resolve by type; nullptr when not registered.
    template <typename T>
    T* find()
    {
        const auto key = std::type_index(typeid(T));
        for (auto& [k, sub] : m_subsystems)
            if (k == key)
                return static_cast<T*>(sub->implPtr());
        return nullptr;
    }

    template <typename T>
    bool has() const
    {
        const auto key = std::type_index(typeid(T));
        for (auto& [k, sub] : m_subsystems)
            if (k == key)
                return true;
        return false;
    }

    /// Per-frame hook dispatch (registration order).
    void beginFrame(f32 dt);
    void endFrame();

    /// onShutdown every subsystem in reverse registration order, then
    /// clear the registry. Safe to call again later (re-register anew).
    void shutdown();

private:
    /// Shut an existing entry with the same key down, onInit the new one,
    /// append; returns the new implementation pointer.
    void* adopt(std::type_index key, std::shared_ptr<ISubsystem> sub);

    std::vector<std::pair<std::type_index, std::shared_ptr<ISubsystem>>> m_subsystems;
};

KENSHIN_NS_END
