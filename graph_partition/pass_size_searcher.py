

class BasePassSizeSearcher:
    """
    Base class for pass size searcher.
    
    Every iteration, the pass_size_limit increases by a ratio (default increase_ratio=1.5).
    """
    
    def __init__(self,
        lower_bound=1 * 2**20, # 1MB
        upper_bound=50 * 2**20, # 50MB
        max_cut_offset_ratio=0.02,
        increase_ratio=1.5,
    ) -> None:
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_cut_offset_ratio = max_cut_offset_ratio
        self.increase_ratio = increase_ratio


    def get_max_cut_offset(self, num_cuttable_nodes: int) -> int:
        """
        get the maximum offset ratio of the cut point

        Args:
            num_cuttable_nodes (int): the number of cuttable nodes (or layers) in the model

        Returns:
            int: max_cut_offset
        """
        return max(1, int(self.max_cut_offset_ratio * num_cuttable_nodes))


    def iter(self):
        """
        iter the max_pass_size

        Yields:
            int: current max_pass_size
        """
        
        current = self.lower_bound
        while current < self.upper_bound:
            yield current
            current = int(current * self.increase_ratio)
        yield self.upper_bound