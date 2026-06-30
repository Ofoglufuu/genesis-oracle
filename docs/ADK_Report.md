# ADK Report

ADK's native state-tracking and tool-calling capabilities significantly simplify agent development compared to the manual while-loops and raw JSON parsing used in Week 9. Instead of explicitly controlling every reasoning and execution step, the framework automatically manages conversation context, tool invocation, result inspection, and retries. This makes the agent architecture cleaner, more reliable, and easier to extend while reducing the amount of fragile orchestration code.
