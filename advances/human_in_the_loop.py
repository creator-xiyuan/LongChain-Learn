"""
人机交互 (HITL)
"""

from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@tool
def write_file_tool(path: str, content: str) -> str:
    """写文件。"""
    return f"已将内容写入文件: {path}"


@tool
def execute_sql_tool(query: str) -> str:
    """执行 SQL。"""
    return f"已执行 SQL: {query}"


@tool
def read_data_tool(path: str) -> str:
    """读取数据。"""
    return f"已读取数据文件: {path}"


agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 写文件：允许 approve / edit / reject
                "write_file_tool": True,
                # 执行 SQL：只允许 approve / reject，不允许 edit
                "execute_sql_tool": {"allowed_decisions": ["approve", "reject"]},
                # 读数据：安全操作，直接通过
                "read_data_tool": False,
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

"""在终端里把中断信息打印清楚，方便你做决策。"""
def pretty_print_interrupt(interrupt: Any) -> None:
    # docs 里的结构：result["__interrupt__"] 是一个 list，元素是 Interrupt(...)
    item = interrupt[0]
    value = getattr(item, "value", item)  # 兼容直接就是 dict 的情况
    action_requests: List[Dict[str, Any]] = value.get("action_requests", [])
    review_configs: List[Dict[str, Any]] = value.get("review_configs", [])

    print("\n=== 检测到需要人工审批的工具调用 ===")
    for idx, action in enumerate(action_requests):
        cfg = review_configs[idx] if idx < len(review_configs) else {}
        allowed = cfg.get("allowed_decisions", ["approve", "edit", "reject"])
        print(f"\n[#{idx}] 待审批工具:")
        print(f"  工具名: {action.get('name')}")
        print(f"  参数:  {action.get('arguments')}")
        print(f"  允许的决策: {allowed}")
        desc = action.get("description")
        if desc:
            print("  描述:")
            print(desc)

"""从命令行收集你对每个工具调用的决策。"""
def collect_decisions(interrupt: Any) -> List[Dict[str, Any]]:
    item = interrupt[0]
    value = getattr(item, "value", item)
    action_requests: List[Dict[str, Any]] = value.get("action_requests", [])
    review_configs: List[Dict[str, Any]] = value.get("review_configs", [])

    decisions: List[Dict[str, Any]] = []
    for idx, action in enumerate(action_requests):
        cfg = review_configs[idx] if idx < len(review_configs) else {}
        allowed = cfg.get("allowed_decisions", ["approve", "edit", "reject"])

        while True:
            choice = input(
                f"\n请为第 {idx} 个工具调用选择决策 {allowed} 之一: "
            ).strip() or "approve"
            if choice in allowed:
                break
            print(f"非法输入，请输入 {allowed} 之一。")

        if choice == "approve":
            decisions.append({"type": "approve"})
        elif choice == "reject":
            msg = input("请输入拒绝原因（可选，回车跳过）: ").strip()
            decision: Dict[str, Any] = {"type": "reject"}
            if msg:
                decision["message"] = msg
            decisions.append(decision)
        elif choice == "edit":
            # 这里只做一个简单示例：允许你直接修改参数字典字符串
            # 实际项目里建议做更细致的字段级编辑。
            print("原始参数: ", action.get("arguments"))
            edited_sql = input("请输入新的 SQL（直接回车沿用原来的）: ").strip()
            args = dict(action.get("arguments") or {})
            if edited_sql:
                # 针对 execute_sql_tool 的参数名是 query
                if "query" in args:
                    args["query"] = edited_sql
                else:
                    args = {"query": edited_sql}
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": action.get("name"),
                        "args": args,
                    },
                }
            )
        else:
            # 理论上不会到这里
            decisions.append({"type": "approve"})

    return decisions


def main() -> None:
    config = {"configurable": {"thread_id": "hitl-demo-thread"}}

    # 第一次运行：可能会触发中断
    result = agent.invoke(
        {
            "messages": [
                HumanMessage("帮我修改/Users/PyCharmProjects/LongChain-Learn-main/quickstart/quickstart.py文件，使其格式化"),
            ]
        },
        config=config,
    )

    if "__interrupt__" not in result:
        print("\n没有触发任何需要人工审批的工具调用。")
        print("Agent 最终返回：")
        print(result)
        return

    interrupt = result["__interrupt__"]
    pretty_print_interrupt(interrupt)
    decisions = collect_decisions(interrupt)

    # 带着你的决策继续执行
    print("\n=== 带着你的决策继续执行 ===")
    final_result = agent.invoke(
        Command(
            resume={
                "decisions": decisions,
            }
        ),
        config=config,
    )

    print("\n=== 最终结果 ===")
    print(final_result)


if __name__ == "__main__":
    main()