# First, let's create the complete executor with performance tracking integration

"""
Graph Executor - Runs agent graphs.

The executor:
1. Takes a GraphSpec and Goal
2. Initializes shared memory
3. Executes nodes following edges
4. Records all decisions to Runtime
5. Returns the final result
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from framework.graph.edge import GraphSpec
from framework.graph.goal import Goal
from framework.graph.node import (
    FunctionNode,
    LLMNode,
    NodeContext,
    NodeProtocol,
    NodeResult,
    NodeSpec,
    RouterNode,
    SharedMemory,
)
from framework.graph.output_cleaner import CleansingConfig, OutputCleaner
from framework.graph.performance_tracker import PerformanceTracker
from framework.graph.validator import OutputValidator
from framework.llm.provider import LLMProvider, Tool
from framework.runtime.core import Runtime


@dataclass
class ExecutionResult:
    """Result of executing a graph."""

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    steps_executed: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    path: list[str] = field(default_factory=list)  # Node IDs traversed
    paused_at: str | None = None  # Node ID where execution paused for HITL
    session_state: dict[str, Any] = field(default_factory=dict)  # State to resume from


class GraphExecutor:
    """Executes agent graphs with optional performance tracking."""

    def __init__(
        self,
        runtime: Runtime,
        llm: LLMProvider | None = None,
        tools: list[Tool] | None = None,
        tool_executor: Callable | None = None,
        node_registry: dict[str, NodeProtocol] | None = None,
        approval_callback: Callable | None = None,
        cleansing_config: CleansingConfig | None = None,
        enable_performance_tracking: bool = True,
    ):
        """
        Initialize the executor.

        Args:
            runtime: Runtime for decision logging
            llm: LLM provider for LLM nodes
            tools: Available tools
            tool_executor: Function to execute tools
            node_registry: Custom node implementations by ID
            approval_callback: Optional callback for human-in-the-loop approval
            cleansing_config: Optional output cleansing configuration
            enable_performance_tracking: Whether to track performance metrics
        """
        self.runtime = runtime
        self.llm = llm
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.node_registry = node_registry or {}
        self.approval_callback = approval_callback
        self.validator = OutputValidator()
        self.logger = logging.getLogger(__name__)

        # Initialize performance tracking
        self.enable_performance_tracking = enable_performance_tracking
        if self.enable_performance_tracking:
            self.performance_tracker = PerformanceTracker(enable_memory_tracking=False)

        # Initialize output cleaner
        self.cleansing_config = cleansing_config or CleansingConfig()
        self.output_cleaner = OutputCleaner(
            config=self.cleansing_config,
            llm_provider=llm,
        )

    def _validate_tools(self, graph: GraphSpec) -> list[str]:
        """
        Validate that all tools declared by nodes are available.

        Returns:
            List of error messages (empty if all tools are available)
        """
        errors = []
        available_tool_names = {t.name for t in self.tools}

        for node in graph.nodes:
            if node.tools:
                missing = set(node.tools) - available_tool_names
                if missing:
                    available = (
                        sorted(available_tool_names) if available_tool_names else "none"
                    )
                    errors.append(
                        f"Node '{node.name}' (id={node.id}) requires tools "
                        f"{sorted(missing)} but they are not registered. "
                        f"Available tools: {available}"
                    )

        return errors

    def _estimate_tokens_used(self, result: NodeResult) -> Optional[int]:
        """Estimate tokens used by a node."""
        # This is a basic estimation - can be enhanced later
        if hasattr(result, 'llm_response') and result.llm_response:
            # Extract token count from LLM response if available
            return getattr(result.llm_response, 'usage', {}).get('total_tokens')
        return None

    async def execute(
        self,
        graph: GraphSpec,
        goal: Goal,
        input_data: dict[str, Any] | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute a graph for a goal.

        Args:
            graph: The graph specification
            goal: The goal driving execution
            input_data: Initial input data
            session_state: Optional session state to resume from (with paused_at, memory, etc.)

        Returns:
            ExecutionResult with output and metrics
        """
        # Start performance tracking
        if self.enable_performance_tracking:
            graph_id = getattr(graph, 'id', 'unknown_graph')
            self.performance_tracker.start_graph(graph_id)
        
        start_time = time.time()

        # Validate graph
        errors = graph.validate()
        if errors:
            return ExecutionResult(
                success=False,
                error=f"Invalid graph: {errors}",
            )

        # Validate tool availability
        errors = self._validate_tools(graph)
        if errors:
            return ExecutionResult(
                success=False,
                error=f"Tool validation failed: {', '.join(errors)}",
            )

        # Initialize memory
        from framework.runtime.shared_state import IsolationLevel, StateScope
        
        execution_id = f"exec_{int(time.time() * 1000)}"
        stream_id = goal.id or "default_stream"
        
        memory = SharedMemory(
            manager=self.runtime.state_manager,
            execution_id=execution_id,
            stream_id=stream_id,
            isolation=IsolationLevel.ISOLATED,
        )

        # Initialize input data
        if input_data:
            for key, value in input_data.items():
                memory.write(key, value)

        # Resume from session state if provided
        if session_state:
            for key, value in session_state.get('memory', {}).items():
                memory.write(key, value)
            paused_at = session_state.get('paused_at')
        else:
            paused_at = None

        # Determine start node
        start_node_id = graph.get_start_node_id()
        if not start_node_id:
            return ExecutionResult(
                success=False,
                error="Graph has no start node",
            )

        # Execute nodes
        current_node_id = start_node_id if not paused_at else paused_at
        steps_executed = 0
        total_tokens = 0
        path = []
        paused_at_node = None

        try:
            while current_node_id:
                node_spec = graph.get_node(current_node_id)
                if not node_spec:
                    return ExecutionResult(
                        success=False,
                        error=f"Node '{current_node_id}' not found in graph",
                        steps_executed=steps_executed,
                        path=path,
                    )

                # Track node start
                if self.enable_performance_tracking:
                    self.performance_tracker.start_node(current_node_id, node_spec.node_type)

                # Execute node
                result = await self._execute_node(current_node_id, node_spec, memory, goal)

                # Track node end
                if self.enable_performance_tracking:
                    tokens_used = self._estimate_tokens_used(result)
                    if tokens_used:
                        total_tokens += tokens_used
                    self.performance_tracker.end_node(
                        node_id=current_node_id,
                        tokens_used=tokens_used,
                        success=result.success,
                        error_message=result.error if not result.success else None
                    )

                steps_executed += 1
                path.append(current_node_id)

                # Check for HITL pause
                if (
                    node_spec.node_type == "human_input"
                    and self.approval_callback
                    and result.success
                ):
                    paused_at_node = current_node_id
                    break

                # Determine next node
                next_node_id = self._follow_edges(
                    graph, goal, current_node_id, node_spec, result, memory
                )

                if not next_node_id:
                    break

                current_node_id = next_node_id

            # End performance tracking
            if self.enable_performance_tracking:
                performance_data = self.performance_tracker.end_graph()
                self.logger.info(f"Performance summary: {performance_data.get_performance_summary()}")

            total_latency_ms = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                success=True,
                output=memory.read_all(),
                steps_executed=steps_executed,
                total_tokens=total_tokens,
                total_latency_ms=total_latency_ms,
                path=path,
                paused_at=paused_at_node,
                session_state={
                    'memory': memory.read_all(),
                    'paused_at': paused_at_node,
                } if paused_at_node else None,
            )

        except Exception as e:
            # End performance tracking even on error
            if self.enable_performance_tracking:
                self.performance_tracker.end_graph()
            
            self.logger.error(f"Graph execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                steps_executed=steps_executed,
                path=path,
            )

    # ... rest of existing methods will be here

# Let me now add the rest of the existing methods
