// KEngineDemo 入口（S0.2：日志接入版）。
// 后续阶段：参数解析（--frames/--size/--screenshot/--cam）→ 引擎初始化 → demo 场景。

#include "ke/Version.h"
#include "ke/core/Log.h"

int main()
{
    ke::log::init();

    KE_INFO("{}", ke::getBanner());

    ke::log::shutdown();
    return 0;
}
