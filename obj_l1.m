function [f_obj,f_grad] = obj_l1(v)
Tf=3;
Size=54;%(要根据数据更改)
m_1= Size;%三折交叉验证中1份的数据集大小（验证集）
m_2=Size*2; %三折交叉验证中2份的数据集大小（训练集）
Tfm1=Tf*m_1;
Tfm2=Tf*m_2;
T=1/ (3*m_1)*[ones(Tfm1,1);zeros(Tfm1+2*Tfm2,1)];
objco=[0,T',0];
f_obj=objco*v;
f_grad=[0;T;0];
end



