#This should be a class to time everything in the vision pipeline and save them to enable
#benchmarking and identifying bottlenecks. It should be used as a context manager, e.g.:
#It should construct a table with the following columns:
# document_name, page_number, step_name, time_taken_seconds
#and save it as a CSV file at the end of the pipeline.
import time
import pandas as pd


class Timer:
    def __init__(self):
        self.records = []

    def time_step(self, document_name: str, page_number: int, step_name: str):
        return self._TimerContext(self, document_name, page_number, step_name)

    class _TimerContext:
        def __init__(self, timer, document_name, page_number, step_name):
            self.timer = timer
            self.document_name = document_name
            self.page_number = page_number
            self.step_name = step_name

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            time_taken = end_time - self.start_time
            self.timer.records.append({
                "document_name": self.document_name,
                "page_number": self.page_number,
                "step_name": self.step_name,
                "time_taken_seconds": time_taken
            })

    def save_to_csv(self, file_path: str):
        df = pd.DataFrame(self.records)
        overview1, overview2, overview3 = self.save_overview(df)
        df.to_csv(file_path, index=False)
        overview1.to_csv(file_path.replace(".csv", "_overview1.csv"), index=False)
        overview2.to_csv(file_path.replace(".csv", "_overview2.csv"), index=False)
        overview3.to_csv(file_path.replace(".csv", "_overview3.csv"), index=False)

    def save_overview(self, df):
        # Create a pivot table to show average time taken per page per document regardless of step
        overview1 = df.pivot_table(
            index=["document_name", "page_number"],
            values="time_taken_seconds",
            aggfunc="mean"
        ).reset_index()

        # Create a pivot table to show average time taken per step across all documents and pages
        overview2 = df.groupby("step_name")["time_taken_seconds"].mean().reset_index()

        # create a third overview that shows the total time taken per document
        overview3 = df.groupby("document_name")["time_taken_seconds"].sum().reset_index
        return overview1, overview2, overview3