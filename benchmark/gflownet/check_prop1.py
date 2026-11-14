from models import *
def legacy(req, seed):
    con=[(p,u) for u,ps in req.items() for p in ps]; pre=con
    return Model(con,pre)
cases={
 "SNARE typed": (Model(SNARE_CON,SNARE_PRE),"VAMP2",[("SNAP25","Syntaxin-1A")]),
 "VAMP2 legacy": (legacy(VAMP2_REQ,"VAMP2"),"VAMP2",[("SNAP25","Syntaxin-1A")]),
 "eIF3 typed": (Model(EIF3_CON,EIF3_PRE),"Prt1",[("Tif34","Tif35")]),
 "eIF3 legacy": (legacy({"Tif34":{"Prt1"},"Tif35":{"Tif34"},"Tif32":{"Prt1"},"Nip1":{"Tif32"}},"Prt1"),"Prt1",[("Tif34","Tif35")]),
}
for name,(m,seed,q) in cases.items():
    T=m.trees(); sb=m.mask([seed])
    cat=sum(is_seed_caterpillar(t,sb) for t in T)
    so_c=seeded_orders(m,seed,True); so_p=seeded_orders(m,seed,False)
    qm=m.mask(q[0]); k=sum(qm in nodes(t) for t in T)
    print(f"{name:13} trees={len(T):5}  seed-caterpillars={cat:4}  seeded(contact)={so_c:4}  seeded(prereq-only)={so_p:4}  query {q[0]}: {k}/{len(T)}")
