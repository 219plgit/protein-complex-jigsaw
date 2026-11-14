from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TypedAssemblyModel:
    """Minimal typed contact/prerequisite model used by PCJ Flow.

    Contacts are undirected pairs. Prerequisites are directed pairs (a,b),
    interpreted as a must be present before b can be completed in a subcomplex.
    """
    units: tuple[str, ...]
    contacts: tuple[tuple[str, str], ...]
    prerequisites: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        idx = {u:i for i,u in enumerate(self.units)}
        if len(idx) != len(self.units):
            raise ValueError("units must be unique")
        for a,b in self.contacts + self.prerequisites:
            if a not in idx or b not in idx:
                raise ValueError(f"unknown component in relation {(a,b)}")
        object.__setattr__(self, "idx", idx)
        n=len(self.units)
        con=[0]*n; pre=[0]*n
        for a,b in self.contacts:
            i,j=idx[a],idx[b]; con[i]|=1<<j; con[j]|=1<<i
        for a,b in self.prerequisites:
            pre[idx[b]] |= 1<<idx[a]
        object.__setattr__(self, "con", tuple(con))
        object.__setattr__(self, "pre", tuple(pre))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "full", (1<<n)-1)

    def mask(self, names):
        return sum(1<<self.idx[u] for u in names)

    def names(self, mask:int):
        return tuple(u for i,u in enumerate(self.units) if mask>>i & 1)

    def connected(self, S:int):
        if not S: return False
        bit=S & -S; seen=bit; stack=[bit.bit_length()-1]
        while stack:
            i=stack.pop(); nb=self.con[i] & S & ~seen
            while nb:
                b=nb & -nb; j=b.bit_length()-1
                seen |= b; stack.append(j); nb ^= b
        return seen == S

    def closed(self, S:int):
        m=S
        while m:
            b=m & -m; i=b.bit_length()-1
            if self.pre[i] & ~S: return False
            m ^= b
        return True

    def merge_ok(self, A:int, B:int):
        linked=any(self.con[i] & B for i in range(self.n) if A>>i & 1)
        return bool(linked and self.closed(A|B))

    def splits(self, S:int):
        low=S & -S; sub=(S-1)&S
        while sub:
            if sub & low:
                A=sub; B=S^sub
                if B and self.connected(A) and self.connected(B) and self.merge_ok(A,B):
                    yield A,B
            sub=(sub-1)&S


def eif3_model():
    return TypedAssemblyModel(
        units=("Prt1","Tif32","Nip1","Tif34","Tif35"),
        contacts=(("Prt1","Tif32"),("Tif32","Nip1"),("Prt1","Tif34"),("Tif34","Tif35")),
        prerequisites=(("Tif34","Tif35"),),
    )


def snare_model():
    units=("VAMP2","SNAP25","Syntaxin-1A","Munc18-1","Complexin","Syt-1","Synaptophysin","SNCA")
    contacts=(("VAMP2","SNAP25"),("VAMP2","Syntaxin-1A"),("VAMP2","Complexin"),
              ("SNAP25","Syntaxin-1A"),("SNAP25","Syt-1"),("Syntaxin-1A","Complexin"),
              ("Syntaxin-1A","Syt-1"),("Syntaxin-1A","Munc18-1"),
              ("VAMP2","Synaptophysin"),("VAMP2","SNCA"))
    prerequisites=(("VAMP2","Complexin"),("Syntaxin-1A","Complexin"),
                   ("SNAP25","Syt-1"),("Syntaxin-1A","Syt-1"),("Syntaxin-1A","Munc18-1"))
    return TypedAssemblyModel(units,contacts,prerequisites)
