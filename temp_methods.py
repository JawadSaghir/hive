# Add this helper method to GraphExecutor class

def _estimate_tokens_used(self, result: NodeResult) -> Optional[int]:
    """Estimate tokens used by a node."""
    # This is a basic estimation - can be enhanced later
    if hasattr(result, 'llm_response') and result.llm_response:
        # Extract token count from LLM response if available
        return getattr(result.llm_response, 'usage', {}).get('total_tokens')
    return None

# Now modify the execute method around line 100
# Add these imports at the top of the file after the others

import time
from framework.graph.performance_tracker import PerformanceTracker
