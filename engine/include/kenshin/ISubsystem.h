#pragma once

// ── ISubsystem — subsystem wrapper + lifecycle interface ─────────
// Concrete implementations do NOT inherit anything: Subsystem<T> holds
// them by value and calls the optional hooks, detected with concepts:
//   void setOwner(Engine*)    — store the engine for cross-subsystem lookups
//   bool onInit(Engine&)      — engine-dependent setup (throw / return false to fail)
//   void onShutdown()         — released in REVERSE registration order
//   void onBeginFrame(f32 dt) / void onEndFrame() — per-frame hooks
// A class with none of these still registers fine (pure service object).

#include "kenshin/Common.h"

#include <concepts>
#include <memory>
#include <utility>

KENSHIN_NS_BEGIN

class Engine;

class ISubsystem
{
public:
    virtual ~ISubsystem() = default;

    virtual void* implPtr() = 0;
    virtual bool onInit(Engine& engine) = 0;
    virtual void onShutdown() {}
    virtual void onBeginFrame(f32 /*dt*/) {}
    virtual void onEndFrame() {}
};

// Optional-hook detection (C++20 concepts instead of SFINAE overloads).
template <typename T>
concept SubsystemHasSetOwner = requires(T& t, Engine* e) { t.setOwner(e); };
template <typename T>
concept SubsystemHasOnInit = requires(T& t, Engine& e) { { t.onInit(e) } -> std::convertible_to<bool>; };
template <typename T>
concept SubsystemHasOnShutdown = requires(T& t) { t.onShutdown(); };
template <typename T>
concept SubsystemHasOnBeginFrame = requires(T& t, f32 dt) { t.onBeginFrame(dt); };
template <typename T>
concept SubsystemHasOnEndFrame = requires(T& t) { t.onEndFrame(); };

/// Holds a subsystem implementation by value; forwards the optional hooks.
template <typename T>
class Subsystem : public ISubsystem
{
public:
    template <typename... Args>
    explicit Subsystem(Args&&... args) : m_impl(std::forward<Args>(args)...) {}

    void* implPtr() override { return &m_impl; }

    bool onInit(Engine& engine) override
    {
        if constexpr (SubsystemHasSetOwner<T>)
            m_impl.setOwner(&engine);
        if constexpr (SubsystemHasOnInit<T>)
        {
            return m_impl.onInit(engine);
        }
        else
        {
            return true;
        }
    }

    void onShutdown() override
    {
        if constexpr (SubsystemHasOnShutdown<T>)
            m_impl.onShutdown();
    }

    void onBeginFrame(f32 dt) override
    {
        if constexpr (SubsystemHasOnBeginFrame<T>)
            m_impl.onBeginFrame(dt);
    }

    void onEndFrame() override
    {
        if constexpr (SubsystemHasOnEndFrame<T>)
            m_impl.onEndFrame();
    }

private:
    T m_impl;
};

/// Holds an already-built instance behind a (possibly abstract) interface
/// pointer — e.g. the render backend from its factory. onShutdown calls
/// impl->shutdown() when the type has one.
template <typename T>
class PointerSubsystem final : public ISubsystem
{
public:
    explicit PointerSubsystem(std::unique_ptr<T> impl) : m_impl(std::move(impl)) {}

    void* implPtr() override { return m_impl.get(); }
    bool onInit(Engine&) override { return true; }

    void onShutdown() override
    {
        if constexpr (requires { m_impl->shutdown(); })
            m_impl->shutdown();
    }

private:
    std::unique_ptr<T> m_impl;
};

KENSHIN_NS_END
