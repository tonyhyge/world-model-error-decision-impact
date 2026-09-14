"""Reproducible thesis illustrations and a deterministically selected real case.

Run with pygame installed (requirements-figures.txt). Conceptual diagrams are
explicit illustrations, not measured results. Existing experiment CSVs unchanged.
"""
import os,sys,json,hashlib,argparse,shutil
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Arc
sys.path.insert(0,str(Path(__file__).resolve().parent))
import figstyle as fs
from experiments.common import ROOT,MAGNITUDES,SEEDS,read_csv
from src.envs.gridworld_mdp import make_choice_gridworld
from src.planning.dp import value_iteration,expected_discounted_return
from src.corruptions.matched_pairs import generate_matched_error_pair

OUT=ROOT/'figures'
BLUE,ORANGE,GREEN,INK='#0072B2','#D55E00','#009E73','#263238'
fs.use()
# Schematics are drawn at the report text width like every other figure, so
# their labels sit a little below the body size rather than being rescaled.
W=fs.TEXT_WIDTH_IN
BODY,NOTE=8.5,7.6
CREATED=[]
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',facecolor='white')
 plt.close(fig);CREATED.append(name)
def canvas(height=4.):
 fig,ax=plt.subplots(figsize=(W,height),layout='constrained')
 ax.set(xlim=(-1.1,11.1),ylim=(0,6));ax.axis('off');return fig,ax

def box(ax,x,y,w,h,text,color=BLUE):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=.12',facecolor=color+'12',edgecolor=color,lw=1.3))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=BODY,color=INK)
def arrow(ax,a,b,label=None):
 ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=INK,lw=1.4))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,label,ha='center',va='bottom',fontsize=NOTE)

def conceptual():
 fig,ax=canvas(2.5)
 ax.set_ylim(1.82,5.38)
 box(ax,.35,2.15,2.35,1.15,'Tác tử\nchọn hành động')
 box(ax,7.25,2.15,2.40,1.15,'Môi trường thật\nchuyển trạng thái')
 box(ax,3.55,4.15,2.90,1.05,'World model\ndự báo hệ quả',GREEN)

 # Luồng tương tác thật: hai đường ngang, dễ đọc theo chiều trái–phải.
 arrow(ax,(2.82,2.95),(7.12,2.95))
 ax.text(4.97,3.12,r'Hành động $a_t$',ha='center',va='bottom',fontsize=9.5,color=INK)
 arrow(ax,(7.12,2.47),(2.82,2.47))
 ax.text(4.97,2.30,r'Trạng thái $s_{t+1}$, phần thưởng $r_{t+1}$',
         ha='center',va='top',fontsize=9.2,color=INK)

 # Luồng học mô hình và planning: màu xanh lá để tách khỏi vòng tương tác thật.
 ax.annotate('',xy=(6.53,4.43),xytext=(8.10,3.42),
             arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.7,
                             connectionstyle='angle3,angleA=90,angleB=0'))
 # Set clear of the curve rather than centred on it, or the text sits on the arrow.
 ax.text(8.62,4.28,'Dữ liệu\ntương tác',ha='left',va='center',fontsize=NOTE,color=GREEN)
 ax.annotate('',xy=(1.90,3.42),xytext=(3.47,4.43),
             arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.7,
                             connectionstyle='angle3,angleA=180,angleB=90'))
 ax.text(1.38,4.28,'Dự báo để\nlập kế hoạch',ha='right',va='center',fontsize=NOTE,color=GREEN)

 save(fig,'method_rl_loop')
 fig,axes=plt.subplots(1,3,figsize=(W,2.9),layout='constrained')
 for ax,panel,vals in zip(axes,['(a) Mốc tham chiếu','(b) Giữ lựa chọn','(c) Đổi lựa chọn'],[[2,1.8],[2.15,1.95],[1.85,1.95]]):
  ax.bar([0,1],vals,color=[BLUE,ORANGE],width=.48,edgecolor=fs.INK,linewidth=.6,zorder=3)
  ax.set(xticks=[0,1],xticklabels=['Hành động A','Hành động B'],ylim=(0,2.5))
  ax.tick_params(axis='x',which='minor',bottom=False,top=False)
  for i,v in enumerate(vals):ax.text(i,v+.05,f'{v:.2f}'.replace('.',','),ha='center',fontsize=NOTE,zorder=4)
  ax.text(.5,2.33,'Chọn '+('A' if vals[0]>vals[1] else 'B'),ha='center',fontsize=BODY)
  fs.panel_label(ax,'',panel)
 axes[0].set_ylabel('Giá trị hành động (minh họa)')
 save(fig,'method_margin_example')
 fig,ax=canvas(3.5)
 box(ax,.2,2.5,2.1,1.1,'Đầu vào\ns (4) + a (2)')
 for y in [4.5,2.5,.5]:
  box(ax,3.3,y,3.1,.9,'MLP: 64 → 64 → 4',GREEN);arrow(ax,(2.4,3),(3.2,y+.45));arrow(ax,(6.5,y+.45),(7.4,3))
 ax.text(4.85,3.9,'5 mạng khởi tạo độc lập',ha='center',fontsize=9)
 ax.text(4.85,1.8,r'$\vdots$',ha='center',fontsize=15)
 box(ax,7.5,2.4,2.1,1.2,'Trung bình Δs\n' r'$s\prime = s + \Delta s$')
 ax.text(5,-.02,r'Lặp cho $a=0$ và $a=1$ → chấm điểm $1+\gamma W(s\prime)$ → chọn điểm lớn hơn',ha='center',fontsize=9)
 save(fig,'method_ensemble')
 fig,ax=canvas(3.6)
 box(ax,.3,4.5,2.7,1,'Tương tác thật\n' r'$(s, a, r, s\prime, \mathrm{done})$')
 box(ax,4,4.5,2.6,1,'Cập nhật Q\ntừ dữ liệu thật');arrow(ax,(3.1,5),(3.9,5))
 box(ax,.3,2.6,2.7,1,'Dyna-Q: lưu mẫu cuối\ncho mỗi (s, a)',GREEN);arrow(ax,(1.65,4.4),(1.65,3.7))
 box(ax,4,2.6,2.6,1,'Chọn transition\ntrong bộ nhớ',GREEN);arrow(ax,(3.1,3.1),(3.9,3.1))
 box(ax,7.2,2.6,2.4,1,'Thêm nhiễu\nsuccessor',ORANGE);arrow(ax,(6.7,3.1),(7.1,3.1))
 box(ax,4,.6,3.9,1,'Cập nhật Q từ mô hình\n5 bước planning / tương tác',GREEN);arrow(ax,(8.4,2.5),(7.3,1.7))
 ax.text(8.3,5,'Q-learning\nkết thúc tại đây',ha='center',fontsize=10)
 ax.text(1.7,1,'done = True:\ntarget chỉ bằng r',ha='center',fontsize=9)
 save(fig,'method_baselines')

def gridplot(ax,env,pi=None,changed=(),target=None,panel=''):
 ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),aspect='equal',xticks=range(5),yticks=range(5),ylabel='Hàng')
 # A cell map has no continuous axis, so minor ticks would be meaningless.
 ax.minorticks_off();ax.tick_params(direction='out',top=False,right=False)
 ax.grid(False)
 fs.panel_label(ax,'Cột',panel)
 for s in range(25):
  r,c=divmod(s,5);color='#FFFFFF'
  if (r,c) in env.hazards:color='#F6D5C4'
  if s==0:color='#D8EDF8'
  if s==24:color='#D6EEE5'
  ax.add_patch(Rectangle((c-.5,r-.5),1,1,facecolor=color,edgecolor='#CCCCCC',lw=.8))
  label='S' if s==0 else 'G' if s==24 else 'H' if (r,c) in env.hazards else ''
  ax.text(c-.33,r-.25,label,fontsize=9,fontweight='bold')
  if pi is not None and s!=24:ax.text(c,r+.1,['↑','↓','←','→'][pi[s]],ha='center',va='center',fontsize=19,color=ORANGE if s in changed else INK)
  if s in changed:ax.add_patch(Rectangle((c-.46,r-.46),.92,.92,fill=False,edgecolor=ORANGE,lw=2))
  if s==target:ax.plot(c,r,'o',ms=28,mfc='none',mec=BLUE,mew=1.5)
 for spine in ax.spines.values():spine.set_visible(False)

def grid_and_case():
 env=make_choice_gridworld();v,q,pi=value_iteration(env)
 fig,axes=plt.subplots(1,2,figsize=(W,3.4),gridspec_kw={'width_ratios':[1,1]},layout='constrained')
 gridplot(axes[0],env,panel='(a) Bố cục môi trường')
 ax=axes[1];ax.axis('off')
 # Fixed-height text against a fractional anchor: keep the blocks far enough
 # apart that they cannot collide when the axes shrinks for the caption.
 ax.text(.03,.97,'S: xuất phát (0, 0)\nG: đích hấp thụ (4, 4)\nH: ô phạt, vẫn đi qua được',
         transform=ax.transAxes,va='top',linespacing=1.7,fontsize=BODY)
 ax.text(.03,.64,'Ví dụ chọn đi lên ở ô bên trong:',transform=ax.transAxes,
         va='top',fontsize=BODY)
 for dx,dy,label in [(0,.20,'0,85'),(-.24,0,'0,075'),(.24,0,'0,075')]:
  ax.annotate('',xy=(.55+dx,.30+dy),xytext=(.55,.30),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.8,color=BLUE))
  ax.text(.55+dx,.30+dy+.035,label,transform=ax.transAxes,ha='center',fontsize=NOTE)
 ax.text(.03,.14,'Đụng biên: giữ vị trí.\nÔ phạt không phải tường chắn.',
         transform=ax.transAxes,va='top',linespacing=1.6,fontsize=BODY)
 # An axis('off') panel cannot show an xlabel, so its caption goes on the
 # figure, on the same baseline as the caption of panel (a).
 fig.text(.76,.012,'(b) Quy tắc chuyển trạng thái',ha='center',va='bottom',
          fontsize=fs.SMALL_PT,color=fs.INK)
 save(fig,'environment_gridworld')
 # Reconstruct the exact saved RNG sequence; first up-branch with local no-flip and nonzero loss.
 # Mirror exp1's draw order exactly, or the replayed stream desynchronises and
 # the reconstructed pair stops matching the row it claims to illustrate.
 saved=read_csv('exp1_matched_pairs.csv');chosen=None
 for seed in SEEDS:
  rng=np.random.default_rng(seed)
  for s in range(24):
   for mag in MAGNITUDES:
    for rep in range(5):rng.dirichlet(np.ones(25))
   for mag,rep in [(m,r) for m in MAGNITUDES for r in range(5)]:
    down,up=generate_matched_error_pair(env,s,int(pi[s]),mag,rng)
    if chosen is None and up.pi_model[s]==pi[s] and up.compute_policy_loss()>1e-9:
     chosen=(seed,s,mag,rep,down,up)
 seed,s,mag,rep,down,up=chosen
 row=next(r for r in saved if int(r['seed'])==seed and int(r['state'])==s
          and float(r['requested_l1'])==mag and int(r['rep'])==rep)
 assert np.isclose(up.compute_policy_loss(),float(row['loss_up']),atol=1e-12)
 changed=np.flatnonzero(pi!=up.pi_model).tolist()
 info={'selection':'first seed/state/repetition in protocol order with up-branch local no-flip and loss > 1e-9; illustrative, not representative',
       'seed':seed,'state':s,'requested_l1':mag,'rep':rep,'branch':'up','changed_states':changed,'l1':up.error.error_l1,'loss':up.compute_policy_loss(),
       'original_policy':pi.tolist(),'model_policy':up.pi_model.tolist(),
       'delta':up.error.delta_p.tolist(),'csv_sha256':hashlib.sha256((ROOT/'results/exp1_matched_pairs.csv').read_bytes()).hexdigest()}
 (OUT/'case_study.json').write_text(json.dumps(info,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(W,3.7),layout='constrained')
 gridplot(axes[0],env,pi,target=s,panel='(a) Chính sách tối ưu thật')
 gridplot(axes[1],env,up.pi_model,changed,s,panel='(b) Chính sách từ mô hình bị nhiễu')
 fig.supxlabel('Vòng xanh: ô can thiệp; viền cam: ô đổi hành động.  Loss = '
                +f'{info["loss"]:.6f}'.replace('.',','),fontsize=fs.SMALL_PT)
 save(fig,'case_gridworld_policy')
 p=env.transitions[s,int(pi[s])];support=np.flatnonzero(p>1e-12)
 fig,ax=plt.subplots(figsize=(W,3.1),layout='constrained');x=np.arange(len(support));w=.23
 for i,(vals,label,color) in enumerate([(down.corrupted_mdp.transitions[s,int(pi[s])],'Chiếu âm',ORANGE),(p,'Phân phối gốc',BLUE),(up.corrupted_mdp.transitions[s,int(pi[s])],'Chiếu dương',GREEN)]):
  ax.bar(x+(i-1)*w,vals[support],w,label=label,color=color,edgecolor=fs.INK,linewidth=.5,zorder=3)
 ax.set(xticks=x,xticklabels=[str(divmod(int(a),5)) for a in support],xlabel='Trạng thái kế tiếp (hàng, cột)',ylabel='Xác suất',ylim=(0,1))
 ax.tick_params(axis='x',which='minor',bottom=False,top=False)
 fs.legend(ax,ncol=3,loc='upper center');save(fig,'method_reflection_pair')
 print(json.dumps(info,indent=2))

def cartpole():
 import gymnasium as gym
 env=gym.make('CartPole-v1',render_mode='rgb_array');env.reset(seed=11)
 state=np.array([.2,0.,.12,0.]);env.unwrapped.state=state.copy()
 frame=env.render();env.close()
 plt.imsave(OUT/'cartpole_render.png',frame)
 fig,axes=plt.subplots(1,2,figsize=(W,3.2),layout='constrained')
 # Gymnasium returns a wide frame that is mostly empty. The ground line spans
 # its full width, so cropping on any dark pixel keeps everything; select the
 # columns carrying real vertical mass (cart and pole) instead.
 mask=(frame<250).any(axis=2)
 columns=np.flatnonzero(mask.sum(axis=0)>5)
 rows=np.flatnonzero(mask.any(axis=1))
 if len(columns) and len(rows):
  pad=28
  frame=frame[max(rows.min()-pad//2,0):rows.max()+pad//2,
              max(columns.min()-pad,0):columns.max()+pad]
 axes[0].imshow(frame);axes[0].axis('off')
 ax=axes[1];ax.set(xlim=(-2,2),ylim=(-.5,2.5),aspect='equal');ax.axis('off')
 ax.plot([-1.8,1.8],[0,0],color=INK,lw=1);ax.add_patch(Rectangle((-.5,.05),1,.35,facecolor='#D8EDF8',edgecolor=INK))
 for x in [-.3,.3]:ax.add_patch(plt.Circle((x,.02),.08,color=INK))
 ax.plot([0,0],[.4,2.25],ls='--',color='gray');ax.plot([0,.65],[.4,2.1],color=ORANGE,lw=5)
 ax.add_patch(Arc((0,.4),1.1,1.1,theta1=69,theta2=90,color=INK));ax.text(.15,1.05,r'$\theta$',fontsize=14)
 arrow(ax,(-.55,.25),(-1.6,.25));arrow(ax,(.55,.25),(1.6,.25))
 ax.text(-1.2,.5,'$a = 0$',ha='center',fontsize=BODY);ax.text(1.2,.5,'$a = 1$',ha='center',fontsize=BODY)
 ax.text(0,-.35,r'$s=(x,\dot{x},\theta,\dot{\theta})$',ha='center',fontsize=12)
 ax.text(.9,1.9,'Cột',fontsize=BODY);ax.text(0,.20,'Xe',ha='center',fontsize=BODY)
 # Both captions on one baseline, as in every other two-panel figure.
 for x,text in zip([.27,.76],['(a) Render từ Gymnasium CartPole-v1',
                              '(b) Các đại lượng trạng thái']):
  fig.text(x,.012,text,ha='center',va='bottom',fontsize=fs.SMALL_PT,color=fs.INK)
 save(fig,'environment_cartpole')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
 conceptual();grid_and_case();cartpole()
 if args.report:
  for name in CREATED:shutil.copy2(OUT/f'{name}.pdf',args.report/'Figures'/f'{name}.pdf')
 print('Created',len(CREATED),'figures')
