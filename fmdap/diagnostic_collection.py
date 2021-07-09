from collections.abc import Mapping, Iterable
from fmdap.diagnostic_output import DiagnosticDataframe, read_diagnostic


class DiagnosticCollection(Mapping):
    def __init__(self, diagnostics=None, names=None):
        self.diagnostics = {}
        self.names = []
        if diagnostics is not None:
            self.add_diagnostics(diagnostics, names)

    def add_diagnostics(self, diagnostics, names=None):
        # TODO: take single file, folder, files with wildcard or DiagnosticDataframe
        if isinstance(diagnostics, str) or (not isinstance(diagnostics, Iterable)):
            diagnostics = [diagnostics]
            if names is not None:
                names = [names]
        if names is None:
            for diag in diagnostics:
                self._add_single_diagnostics(diag)
        else:
            if len(names) != len(diagnostics):
                raise ValueError("diagnostics and names must have same length!")
            for name, diag in zip(names, diagnostics):
                self._add_single_diagnostics(diag, name=name)

    def _add_single_diagnostics(self, diagnostic, name=None):
        if isinstance(diagnostic, str):
            diagnostic = read_diagnostic(diagnostic, name=name)
        assert isinstance(diagnostic, DiagnosticDataframe)
        if name is None:
            name = diagnostic.name
        if name in self.names:
            raise ValueError(
                f"Cannot add diagnostic with name={name}; already in {self.names}"
            )
        self.names.append(name)
        self.diagnostics[name] = diagnostic

    def __repr__(self):
        out = []
        out.append(f"<{type(self).__name__}>")
        for name in self.names:
            diag = self[name]
            out.append(f" - {name}: {diag.type}")
        return str.join("\n", out)

    def __getitem__(self, x):
        if isinstance(x, int):
            x = self._get_diag_name(x)

        return self.diagnostics[x]

    def _get_diag_name(self, diag):
        return self.names[self._get_diag_id(diag)]

    def _get_diag_id(self, diag):
        if diag is None or len(self) <= 1:
            return 0
        elif isinstance(diag, str):
            if diag in self.names:
                diag_id = self.names.index(diag)
            else:
                raise KeyError(f"diagnostic {diag} could not be found in {self.names}")
        elif isinstance(diag, int):
            if diag >= 0 and diag < len(self):
                diag_id = diag
            else:
                raise IndexError(
                    f"diagnostic id was {diag} - must be within 0 and {len(self)-1}"
                )
        else:
            raise KeyError("diagnostic must be None, str or int")
        return diag_id

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __iter__(self):
        return iter(self.diagnostics.values())

