from models import *
base=Model(SNARE_CON,SNARE_PRE)
plus=Model(SNARE_CON,SNARE_PRE+[("SNAP25","Syntaxin-1A")])
for name,m in [("typed",base),("typed + SNAP25<Syntaxin",plus)]:
    T=m.trees(); q=m.mask(["SNAP25","Syntaxin-1A"])
    print(name, "merger",len(T),"acceptor",sum(q in nodes(t) for t in T),
          "seeded(contact)",seeded_orders(m,"VAMP2",True),"seeded(prereq-only)",seeded_orders(m,"VAMP2",False))
# typed seeded supports of the native-MS queries (caterpillar trees)
sb=base.mask(["VAMP2"]); T=[t for t in base.trees() if is_seed_caterpillar(t,sb)]
for q in [["SNAP25","VAMP2"],["VAMP2","SNAP25","Syntaxin-1A"],["VAMP2","SNAP25","Syntaxin-1A","Complexin"],["VAMP2","Syntaxin-1A"]]:
    mk=base.mask(q); print(q, sum(mk in nodes(t) for t in T),"/",len(T))
