class PrettyChart:
    def __init__(self, title: str = None, field_name: list = None) -> None:
        self.title = title
        self.field_name = field_name
        self.chart = dict()
        
    def __getitem__(self, key):
        return self.chart[key]


class PrettyTable(PrettyChart):
    def __init__(self) -> None:
        super().__init__()
