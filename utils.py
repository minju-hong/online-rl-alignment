import sys
import time

def now_stamp() -> str:
    """Returns a formatted timestamp string."""
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

class ProgressBar:
    """An inline progress bar that overwrites the terminal line using \r"""
    def __init__(self, total: int, prefix: str = ""):
        self.total = max(1, int(total))
        self.prefix = prefix
        self.start = time.perf_counter()
        self.next_pct = 10
        # Print the 0% state immediately
        sys.stdout.write(f"\r{self.prefix} [{' ' * 20}]   0% (0/{self.total}) | elapsed: 0.0s | eta: ?")
        sys.stdout.flush()

    def update(self, done: int):
        pct = int(round(100 * done / self.total))
        
        # Update every 10% or on the final step
        if pct >= self.next_pct or done >= self.total:
            now = time.perf_counter()
            elapsed = now - self.start
            rate = elapsed / done if done > 0 else None
            eta = rate * (self.total - done) if rate is not None else None

            bar_len = 20
            filled = int(round(bar_len * pct / 100))
            bar = "#" * filled + "-" * (bar_len - filled)

            eta_str = f"{eta:.1f}s" if eta is not None else "?"
            
            # \r forces the cursor back to the start of the line
            sys.stdout.write(f"\r{self.prefix} [{bar}] {pct:3d}% ({done}/{self.total}) | elapsed: {elapsed:.1f}s | eta: {eta_str}")
            sys.stdout.flush()

            while self.next_pct <= pct:
                self.next_pct += 10

            if done >= self.total:
                sys.stdout.write("\n") # Lock in the line when finished