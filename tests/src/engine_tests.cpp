// Engine subsystem registry semantics: typed registration/lookup,
// constructor-argument forwarding, optional lifecycle hooks, LIFO
// shutdown, failing onInit.

#include <catch2/catch_test_macros.hpp>

#include <kenshin/Engine.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct PlainService // no hooks at all — registers fine
    {
        int value = 7;
    };

    struct HookedBase // full hook set; ctor arg proves forwarding
    {
        explicit HookedBase(std::string n) : name(std::move(n)) {}
        std::string name;
        static std::vector<std::string>* log;
        kenshin::Engine* owner = nullptr;

        void setOwner(kenshin::Engine* e) { owner = e; }
        bool onInit(kenshin::Engine&) { log->push_back(name + ":init"); return true; }
        void onShutdown() { log->push_back(name + ":shutdown"); }
        void onBeginFrame(kenshin::f32) { log->push_back(name + ":begin"); }
        void onEndFrame() { log->push_back(name + ":end"); }
    };
    std::vector<std::string>* HookedBase::log = nullptr;

    // Two distinct TYPES (the registry keys by type_index — same-type
    // registrations replace each other, see the re-register section).
    struct HookedService : HookedBase { using HookedBase::HookedBase; };
    struct OtherHooked : HookedBase { using HookedBase::HookedBase; };

    struct FailingService
    {
        bool onInit(kenshin::Engine&) { return false; }
    };

    struct IThing { virtual ~IThing() = default; virtual int id() const = 0; };
    struct ThingImpl : IThing { int id() const override { return 42; } };
} // namespace

TEST_CASE("engine subsystem registry", "[engine]")
{
    kenshin::Engine& engine = kenshin::Engine::get();
    engine.shutdown(); // start clean (the singleton may carry prior state)

    std::vector<std::string> log;
    HookedService::log = &log;

    SECTION("by-value registration, typed lookup, has/find")
    {
        auto& svc = engine.registerSubsystem<PlainService>();
        svc.value = 3;
        REQUIRE(engine.get<PlainService>().value == 3);
        REQUIRE(engine.has<PlainService>());
        REQUIRE(engine.find<PlainService>() != nullptr);
        REQUIRE(engine.find<HookedService>() == nullptr);
        engine.shutdown();
    }

    SECTION("hooks: setOwner + onInit fire on register; frame dispatch")
    {
        auto& hooked = engine.registerSubsystem<HookedService>(std::string("A"));
        REQUIRE(hooked.owner == &engine);
        REQUIRE(log == std::vector<std::string>{"A:init"});

        engine.beginFrame(0.016f);
        engine.endFrame();
        REQUIRE(log == std::vector<std::string>({"A:init", "A:begin", "A:end"}));
        engine.shutdown();
    }

    SECTION("shutdown releases in reverse registration order")
    {
        engine.registerSubsystem<HookedService>(std::string("A"));
        engine.registerSubsystem<OtherHooked>(std::string("B"));
        engine.shutdown();
        REQUIRE(log == std::vector<std::string>({"A:init", "B:init",
                                                 "B:shutdown", "A:shutdown"}));
    }

    SECTION("re-registering a key shuts the old instance down first")
    {
        engine.registerSubsystem<HookedService>(std::string("A"));
        engine.registerSubsystem<HookedService>(std::string("A2"));
        REQUIRE(log == std::vector<std::string>({"A:init", "A:shutdown", "A2:init"}));
        engine.shutdown();
    }

    SECTION("pointer registration under an interface key")
    {
        engine.registerInstance<IThing>(std::make_unique<ThingImpl>());
        REQUIRE(engine.get<IThing>().id() == 42);
        engine.shutdown();
    }

    SECTION("get of unregistered subsystem throws")
    {
        REQUIRE_THROWS_AS(engine.get<PlainService>(), std::runtime_error);
    }

    SECTION("failing onInit throws and leaves nothing registered")
    {
        REQUIRE_THROWS_AS(engine.registerSubsystem<FailingService>(),
                          std::runtime_error);
        REQUIRE_FALSE(engine.has<FailingService>());
        engine.shutdown();
    }

    HookedBase::log = nullptr;
}
