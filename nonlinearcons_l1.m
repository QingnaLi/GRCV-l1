function [c,ceq,dc,dceq]=nonlinearcons_l1(v,Size,feature,heart_scale_label,heart_scale_inst)

m_1= Size;%三折交叉验证中1份的数据集大小（验证集）
m_2=Size*2; %三折交叉验证中2份的数据集大小（训练集）

%%
h_lab_1=heart_scale_label(1:Size);%第一份的y，是三折的验证集的y
rep_3=repmat(h_lab_1,1,feature);
h_lab_2=heart_scale_label(Size+1:2*Size);%第二份的y, 是二折的验证集的y
rep_2=repmat(h_lab_2,1, feature);
h_lab_3= heart_scale_label(2*Size+1:3*Size);%第三份的y，是一折的验证集的y
rep_1=repmat(h_lab_3,1, feature);
h_inst_1=heart_scale_inst(1:Size,:); %是三折的验证集的x

h_inst_2=heart_scale_inst(Size+1: 2*Size,:); %是二折的验证集的x
h_inst_3=heart_scale_inst(2*Size+1: 3*Size,:); %是一折的验证集的x
comb_lab_12=[h_lab_1; h_lab_2];%一折的训练集的y
comb_inst_12=[ h_inst_1; h_inst_2];% 一折的训练集的x
comb_lab_13=[h_lab_1; h_lab_3];%二折的训练集的y
comb_inst_13=[ h_inst_1; h_inst_3];% 二折的训练集的x
comb_lab_23=[h_lab_2; h_lab_3];%三折的训练集的y
comb_inst_23=[ h_inst_2; h_inst_3]; %三折的训练集的x
rep_12= repmat(comb_lab_12,1,feature);
rep_13= repmat(comb_lab_13,1,feature);
rep_23= repmat(comb_lab_23,1,feature);
B_1=rep_12.*comb_inst_12;
A_1=rep_1.* h_inst_3;
B_2=rep_13.*comb_inst_13;
A_2=rep_2.* h_inst_2;
B_3=rep_23.* comb_inst_23;
A_3=rep_3.* h_inst_1;
Tf=3;%T_fold
Tfm1=Tf*m_1;
Tfm2=Tf*m_2;
m_hat=2*Tf*(m_1+m_2);
A=[A_1,zeros(m_1,2*feature);zeros(m_1,feature),A_2, zeros(m_1,feature); zeros(m_1,2*feature),A_3];%é??è??é????°???
B=[B_1,zeros(m_2, 2*feature);zeros(m_2, feature),B_2, zeros(m_2,feature);zeros(m_2,2*feature),B_3];
AB=A*B';%?????????é??????§?è?????
BB=B*B';%?????????é??????§?è?????
M=[zeros(2*Tfm1+Tfm2,1);ones(Tfm2,1)];
N=[zeros(Tfm1),eye(Tfm1),AB,zeros(Tfm1,Tfm2);-eye(Tfm1),zeros(Tfm1),zeros(Tfm1,Tfm2), zeros(Tfm1,Tfm2);zeros(Tfm2,Tfm1),zeros(Tfm2,Tfm1),BB,eye(Tfm2);zeros(Tfm2,Tfm1),zeros(Tfm2,Tfm1),-eye(Tfm2),zeros(Tfm2)];%????????????
P=[M,N,zeros(m_hat,1)];%这里多加了一个变量t_k，所以多了最后一列
Q=[zeros(m_hat,1),eye(m_hat),zeros(m_hat,1)];
a=[zeros(Tfm1,1);ones(Tfm1,1);-ones(Tfm2,1);zeros(Tfm2,1)];
c=[(P*v+a).*(v(2:m_hat+1))-v(m_hat+2)*ones(m_hat,1)];
ceq=[];
dc=diag(Q*v)*P+diag(a+P*v)*Q;
% dc(1:m_hat,m_hat+2)=0;
dceq=[];
end

