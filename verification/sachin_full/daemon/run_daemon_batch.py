import subprocess, json, os, sys, time, itertools, signal
P="/tmp/claude-0/-home-claude/3444b933-a437-5e87-b08c-a427ed2d6d1b/scratchpad"
REPO="/home/claude/teraformer09/adaptive-variational-inference-decoder"
BR=f"{P}/bridge_build/avid_cosim_bridge"; DM=f"{P}/sw_patched/b/avid_daemon"; CK=f"{REPO}/checkpoints/r2d2_v1_trained.bin"
SCEN=sys.argv[1].split(","); CTRL=sys.argv[2].split(","); SEEDS=[int(x) for x in sys.argv[3].split(",")]
ROUNDS=int(sys.argv[4]); WIN=int(sys.argv[5]); OUT=sys.argv[6]
rows=[]
for sc,ct,sd in itertools.product(SCEN,CTRL,SEEDS):
    sock=f"/tmp/avid_{os.getpid()}.sock"; mfile=f"{P}/dm_{os.getpid()}.json"
    if os.path.exists(sock): os.remove(sock)
    b=subprocess.Popen([BR,"--socket",sock],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(0.4)
    cmd=[DM,"--backend","sim","--socket",sock,"--scenario",sc,"--controller-mode",ct,"--rounds",str(ROUNDS),"--window",str(WIN),"--seed",str(sd),"--metrics-out",mfile]
    if ct=="avid_r2d2": cmd+=["--checkpoint",CK]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
    b.kill(); b.wait()
    if r.returncode!=0 or not os.path.exists(mfile): print("FAIL",sc,ct,sd,r.returncode,r.stderr[-200:]); continue
    d=json.load(open(mfile)); os.remove(mfile)
    rows.append(dict(scenario=sc,controller=ct,seed=sd,rounds=ROUNDS,static_fail=d["raw_static_b0_failures"],avid_fail=d["raw_avid_z_failures"],ler_static=d["static_b0_logical_error_rate"],ler=d["logical_error_rate"],cycles=d["bp_core_cycles_mean"]))
json.dump(rows,open(OUT,"w"))
print("done",len(rows))
