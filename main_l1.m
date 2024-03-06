clear
clc
setenv('SNOPT_LICENSE','D:\APPs\MATLAB\snopt7.lic')

total_size=270;%(Ҫ�������ݸ���)
Size=54;%(Ҫ�������ݸ���)
[original_label, original_inst] =  libsvmread(['heart.txt']);%(Ҫ�������ݸ���)

heart_scale_label=original_label(1:total_size);
heart_scale_inst=original_inst(1:total_size,:);
feature=size(heart_scale_inst,2);

%% ��������1
if isempty(find(heart_scale_label==1)) %1: no
    if isempty(find(heart_scale_label==-1)) %1: no, -1: no
        heart_scale_label(heart_scale_label==heart_scale_label(1)) = 1; %��==first�ĸĳ�1
        heart_scale_label(heart_scale_label~=1) = -1; %�Ѵ�ʱ��==1�ĸĳ�-1
    else %1: no, -1: yes
        heart_scale_label(heart_scale_label~=-1) = 1; %��~=-1�ĸĳ�1
    end
else %1: yes
    if isempty(find(heart_scale_label==-1)) %1: yes, -1: no
        heart_scale_label(heart_scale_label~=1) = -1; %��~=1�ĸĳ�-1
    end
end

%%
tic
m_1= Size;%���۽�����֤��1�ݵ����ݼ���С����֤����
m_2=Size*2; %���۽�����֤��2�ݵ����ݼ���С��ѵ������
n_tr=Size*3;
n_te=total_size-n_tr;
%���������ݼ���ֻ����l_train.
h_lab_1=heart_scale_label(1:Size);%��һ�ݵ�y�������۵���֤����y
rep_3=repmat(h_lab_1,1,feature);
h_lab_2=heart_scale_label(Size+1:2*Size);%�ڶ��ݵ�y, �Ƕ��۵���֤����y
rep_2=repmat(h_lab_2,1, feature);
h_lab_3= heart_scale_label(2*Size+1:3*Size);%�����ݵ�y����һ�۵���֤����y
rep_1=repmat(h_lab_3,1, feature);
h_inst_1=heart_scale_inst(1:Size,:); %�����۵���֤����x
h_inst_2=heart_scale_inst(Size+1: 2*Size,:); %�Ƕ��۵���֤����x
h_inst_3=heart_scale_inst(2*Size+1: 3*Size,:); %��һ�۵���֤����x
comb_lab_12=[h_lab_1; h_lab_2];%һ�۵�ѵ������y
comb_inst_12=[ h_inst_1; h_inst_2];% һ�۵�ѵ������x
comb_lab_13=[h_lab_1; h_lab_3];%���۵�ѵ������y
comb_inst_13=[ h_inst_1; h_inst_3];% ���۵�ѵ������x
comb_lab_23=[h_lab_2; h_lab_3];%���۵�ѵ������y
comb_inst_23=[ h_inst_2; h_inst_3]; %���۵�ѵ������x
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
A=[A_1,zeros(m_1,2*feature);zeros(m_1,feature),A_2, zeros(m_1,feature); zeros(m_1,2*feature),A_3];
B=[B_1,zeros(m_2, 2*feature);zeros(m_2, feature),B_2, zeros(m_2,feature);zeros(m_2,2*feature),B_3];
AB=A*B';
BB=B*B';
M=[zeros(2*Tfm1+Tfm2,1);ones(Tfm2,1)];
N=[zeros(Tfm1),eye(Tfm1),AB,zeros(Tfm1,Tfm2);-eye(Tfm1),zeros(Tfm1),zeros(Tfm1,Tfm2), zeros(Tfm1,Tfm2);zeros(Tfm2,Tfm1),zeros(Tfm2,Tfm1),BB,eye(Tfm2);zeros(Tfm2,Tfm1),zeros(Tfm2,Tfm1),-eye(Tfm2),zeros(Tfm2)];
T=1/ (3 * m_1)*[ones(Tfm1,1);zeros(Tfm1+2*Tfm2,1)];
P=[M,N,zeros(m_hat,1)];%��������һ������t_k�����Զ������һ��
a=[zeros(Tfm1,1);ones(Tfm1,1);-ones(Tfm2,1);zeros(Tfm2,1)];

%% ���ò���
C_lb=0;%C���½�
lb=[C_lb;zeros(m_hat,1);-inf];
v_01=60;%C�ĳ�ʼֵ
v_0=[v_01;zeros(m_hat,1);1];
[v,f,info,output,lambda,states]=snsolve(@(v_0)obj_l1(v_0),v_0,-P,a,[],[],lb,[],@(v_0)nonlinearcons_l1(v_0,Size,feature,heart_scale_label,heart_scale_inst));  
iter_k=output.iterations;
C=v(1)*1.5;
t=toc;

% G_opt=P*v+a;
% H_opt=v(2:m_hat+1);
% GH=G_opt-H_opt;
% index_1=find(GH>0);
% index_2=find(GH<=0);                                                                                                                                                                                                                                                                                                                                                                                                        
% absH=abs(H_opt(index_1));
% absG=abs(G_opt(index_2));
% combine=[absH;absG];
% maxVio=max(combine);

%% ���Խ��
%% Reformulate the training set
Xtrain =  heart_scale_inst(1:n_tr,:);
Ytrain = heart_scale_label(1:n_tr);
XtrainALM = [Xtrain, ones(n_tr,1)];
B = XtrainALM;
ind = Ytrain == 1;
B(ind,:) = -B(ind,:);
B_T = B';

%% Parameters settings
flag_prnt = 0;
sigma0 = 0.15;
tau = 1.0;

%% Training
Info_in.C = C;
Info_in.sigma0 = sigma0;
Info_in.tau = tau;
[w,~,~,~] = almsncg_SVC(B, B_T, flag_prnt, Info_in);

%% Testing
Xtest  =  heart_scale_inst(n_tr+1:end,:);
Ytest  = heart_scale_label(n_tr+1:end);

N = length(Ytest);
b = w(end);
wr = w(1:end-1);

Ymodel = Xtest * wr + b;
Ymodel = double(Ymodel>0);
Ymodel(Ymodel==0) = -1;

error_rate = mean(Ymodel ~= Ytest)*100;

fprintf('==== Finish Training! ====\n')
fprintf('The error rate of the test set : %.3f\n', error_rate)
fprintf('Time : %.3f\n', t)
fprintf('The final hyperparameter : %.3f\n', C)


