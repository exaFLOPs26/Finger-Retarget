import os; os.environ['MUJOCO_GL']='osmesa'
import mujoco, numpy as np, imageio, imageio.v3 as iio
run="/scratch/exaflops/do-as-i-do/retargeting/outputs/sharpa/right/whisking/0"
spec=mujoco.MjSpec.from_file(f"{run}/scene.xml"); spec.visual.global_.offwidth=1280; spec.visual.global_.offheight=720
model=spec.compile(); data=mujoco.MjData(model)
dyn=[i for i in range(1,model.nbody)]  # all bodies except world
d=np.load(f"{run}/trajectory_mjwp.npz",allow_pickle=True); q=np.asarray(d['qpos'])
if 'sim_step' in d.files and len(d['sim_step'])==q.shape[0]: q=q[np.argsort(np.asarray(d['sim_step']).ravel())]
q=q.reshape(-1,q.shape[-1])[600:]
lo=np.full(3,1e9); hi=np.full(3,-1e9)
for fi in range(0,len(q),4):
    data.qpos[:]=q[fi]; mujoco.mj_kinematics(model,data); p=data.xpos[dyn]
    lo=np.minimum(lo,p.min(0)); hi=np.maximum(hi,p.max(0))
center=(lo+hi)/2; extent=float(np.linalg.norm(hi-lo))
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:]=center; cam.distance=extent*1.05+0.08; cam.azimuth=120; cam.elevation=-20
print("distance",round(cam.distance,3),"lookat",center.round(3))
r=mujoco.Renderer(model,720,1280); idx=list(range(0,len(q),2)); frames=[]
for fi in idx:
    data.qpos[:]=q[fi]; mujoco.mj_kinematics(model,data); r.update_scene(data,camera=cam); frames.append(r.render())
r.close()
frames=np.array(frames)
# verify: coverage + edge clipping per checkpoint
for name,i in [("start",0),("q1",len(frames)//4),("mid",len(frames)//2),("q3",3*len(frames)//4),("end",len(frames)-1)]:
    fr=frames[i]; m=(fr.mean(2)<235); ys,xs=np.where(m)
    if len(xs):
        clip=[]
        if xs.min()<=1:clip.append("L")
        if xs.max()>=1278:clip.append("R")
        if ys.min()<=1:clip.append("T")
        if ys.max()>=718:clip.append("B")
        print(f"  {name}: cover={100*m.mean():.1f}% clip={clip or 'none'}")
    else: print(f"  {name}: EMPTY")
out=f"{run}/whisking_retarget.mp4"
imageio.mimwrite(out, list(frames), fps=30, quality=9, macro_block_size=1)
imageio.mimwrite("/scratch/exaflops/daid_home/whisking_retarget.mp4", list(frames), fps=30, quality=9, macro_block_size=1)
iio.imwrite("/scratch/exaflops/daid_home/_final_mid.png", frames[len(frames)//2])
print("WROTE", out, len(frames),"frames")
