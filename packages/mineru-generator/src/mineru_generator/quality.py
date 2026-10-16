from collections import deque


class QualityMonitor:
    def __init__(self, window_size=15, consecutive_threshold=3):
        self.window = deque(maxlen=window_size)
        self.consecutive_bad = 0
        self.consecutive_threshold = consecutive_threshold
        self.reset_count = 0
        self.total_checks = 0
        self.bad_checks = 0

    def check(self, record):
        self.total_checks += 1
        is_bad = not record.get("parse_success", False) or not record.get("validation_passed", False)
        if is_bad:
            self.bad_checks += 1
            self.consecutive_bad += 1
        else:
            self.consecutive_bad = 0
        self.window.append({
            "parse_success": record.get("parse_success", False),
            "validation_passed": record.get("validation_passed", False),
            "content_length": record.get("content_length", 0),
        })
        if self.consecutive_bad >= self.consecutive_threshold:
            return False
        return True

    def should_reset(self):
        return self.consecutive_bad >= self.consecutive_threshold

    def get_window_stats(self):
        if not self.window:
            return {"total": 0, "bad": 0, "avg_length": 0}
        total = len(self.window)
        bad = sum(1 for r in self.window if not r["parse_success"] or not r["validation_passed"])
        avg_len = sum(r["content_length"] for r in self.window) / total if total > 0 else 0
        return {"total": total, "bad": bad, "avg_length": round(avg_len, 0), "bad_ratio": round(bad / total, 2)}

    def reset(self):
        self.window.clear()
        self.consecutive_bad = 0
        self.reset_count += 1

    def summary(self):
        return {
            "total_checks": self.total_checks,
            "bad_checks": self.bad_checks,
            "reset_count": self.reset_count,
            "consecutive_bad": self.consecutive_bad,
            "bad_ratio": round(self.bad_checks / max(self.total_checks, 1), 3),
        }
