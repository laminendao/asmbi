for j=1:10
    mat(j,j)=0;
end

for j=1:10
   S(j,j)=0;
end


%% affichage des coef de RV entre les blocs:
%nb_bloc est le nombre de bloc
nb_bloc=10;
vect=5*ones(10,1);
% %=====il faut changer selon le nb classe voulu
nb_class=4;
[ all_matrix ] = preparingDataStatis( data10,vect );
for i=1:nb_bloc
    all_matrix(i).matrix=som_normalize(all_matrix(i).matrix,'var');
end
[a S]= SMatrix( all_matrix,2 );



%% ce bout de code donne la moyenne des RV des partitions
res=all_result(1).result;

mat=zeros(10,10);
for i=1:30
    mat=mat+res(i).RV_part;
end
mat=mat/30
%% ce bout de code donne l'ecart type des RV des partitions
res=all_result(7).result;

mat=zeros(10,10);
mat2=zeros(10,10);
for j=1:10
    for k=1:10
        vect=zeros(1,30);
for i=1:30
    mat=res(i).RV_part;
    vect(i)=mat(j,k);
end
mat2(j,k)=std(vect);
    end
end


mat=mat/30
 %% Ce bout de code affiche les poids de STATIS pour les 30 part:
 
 all_poids=zeros(10,10);
 for j=1:10
     res=all_result(j).result;
 
 poids_statis=zeros(10,1);
 for i =1:30
     poids_statis= poids_statis+res(i).poids_statis;
 end
 poids_statis=poids_statis/30;
 all_poids(:,j)=poids_statis;
 end
 %% Ce bout de code affiche les écarts types des poids de CSTATIS pour les 30 part:
 
 all_ecart=zeros(10,10);
 for j=1:10
     res=all_result(j).result;
 
 poids_statis=zeros(10,1);
 vect=zeros(1,30);
 for k=1:10
 for i =1:30
     poids_statis= res(i).poids_statis;%vecteur de 10 elt
     vect(i)=poids_statis(k);
 end
 all_ecart(k,j)=std(vect);
 end
 
 end
 %% code qui affiche la moyenne des RV-LABEL (entre la vraie partition et les partitions données des blocs)
 RV_lab=zeros(10,1);
 for i=1:30
     RV_lab=RV_lab+res(i).RV_Label;
 end
 RV_lab=RV_lab/30;
 
 
 %% code qui affiche les écarts des RV_Lab:
 RV_lab2=zeros(10,1);
 for j=1:10
     vect=zeros(1,30);
 for i=1:30
     RV_lab=res(i).RV_Label;
     vect(i)=RV_lab(j);
 end
 RV_lab2(j)=std(vect);
 end;
 
 %% Ce bout de code affiche les poids de weighted pour les 30 part:
 
 all_poidsW=zeros(10,10);
 for j=1:10
     res=all_result(j).result;
 
 poids_weighted=zeros(10,1);
 for i =1:30
     poids_weighted= poids_weighted+res(i).poidsWeight;
 end
 poids_weighted=poids_weighted/30;
 all_poidsW(:,j)=poids_weighted;
 end
 %% calcul des accuracy des statis:
 all_accStat=zeros(15,1);
 for i=1:15
     res=all_result(i).result;
     acc=0;
     for j=1:30
     acc=acc+res(j).statis_acc;
     end
     acc=acc/30;
     all_accStat(i)=acc;
 end
 %% tableau pour les box-plot:
 all_acc_stat=zeros(30,10);
 for i=1:10
     res=all_result(i).result;
     for j=1:30
         all_acc_stat(j,i)=res(j).statis_acc;
     end
 end
         
 %% tableau de acc de cspa box-plot
 
 all_acc_cspa=zeros(30,10);
 for i=1:10
      res=all_result(i).result;
     for j=1:30
         all_acc_cspa(j,i)=res(j).Cspa_acc;
     end
 end
 %% tableau de acc de weigh box plot
 all_acc_weig=zeros(30,10);
 for i=1:10
     res=all_result(i).result;
     for j=1:30
         all_acc_weig(j,i)=res(j).accweight;
     end
 end
 
 
 %% tableau de acc de tri_Nmf box plot
 all_acc_nmf=zeros(30,10);
 for i=1:10
     res=all_result(i).result;
     for j=1:30
         all_acc_nmf(j,i)=res(j).accNMF;
     end
 end
%% Accuracy weighted

 all_accWei=zeros(15,1);
 for i=1:15
     res=all_result(i).result;
     acc=0;
     for j=1:30
     acc=acc+res(j).accweight;
     end
     acc=acc/30;
     all_accWei(i)=acc;
 end
 %% affichage accuracy CSPA
 
 
 all_accCSPA=zeros(15,1);
 for i=1:15
     res=all_result(i).result;
     acc=0;
     for j=1:30
     acc=acc+res(j).Cspa_acc;
     end
     acc=acc/30;
     all_accCSPA(i)=acc;
 end
 %% affichage des accuracy de NMF
 all_accNMF=zeros(15,1);
 for i=1:15
     res=all_result(i).result;
     acc=0;
     for j=1:30
     acc=acc+res(j).accNMF;
     end
     acc=acc/30;
     all_accNMF(i)=acc;
 end
 
 %% Afficchage des accuracy concaténé
     
%% Moyenne des accuracy des blocs:
% acc_bloc=zeros(10,1);
% for i=1: 30 
%     acc_bloc=acc_bloc+ res(i).acc_Label;
% end
% acc_bloc=acc_bloc/30;
%%calul des RV entre
%% affichage de figure:
Xlab=[1:15];
figure();
    set (gca,'XTick',1:15,'XTicklabel',Xlab);
    hold on;
    plot(all_accStat,'r-*');
    plot(all_accCSPA,'k-*');
    plot(all_accWei,'m-*');
    plot(all_accNMF,'g-*')
    title('moyenne des accuracy pour les 4 algorithmes');
    legend('STATIS','CSPA','Weighted','tri_NMF')




 