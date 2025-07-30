
## Chargement des packages
library(ClustVarLV)
library(MASS)
library(FactoMineR)
library(writexl)
# utilisation de la fonction SPC 
library(PMA)
library(clusterCrit)
library(aricode)
library(corrplot)
library(diceR)
library(readxl)

### Tester les codes de Mory

FctClust=function(ResSAS,Pos1,Pos2){
Res=NULL
Dim=dim(ResSAS)
Clust=rep(0,Dim[1])
for(i in 1:length(Pos1)){
  Clust=rep(0,Dim[1])
  Unik=unique(ResSAS[,Pos1[i]])
  Unik=Unik[-which(is.na(Unik))]
  for(k in 1:(length(Unik)-1)){
    deb=which(ResSAS[,Pos1[i]]==Unik[k])
    fin=which(ResSAS[,Pos1[i]]==Unik[k+1])-1
    PosClust=ResSAS[deb:fin,Pos2[i]]
    Clust[PosClust]=Unik[k]
  }
  deb=which(ResSAS[,Pos1[i]]==Unik[length(Unik)])
  fin=length(ResSAS[,1])
  PosClust=ResSAS[deb:fin,Pos2[i]]
  Clust[PosClust]=Unik[length(Unik)]
  Res=cbind(Res,Clust)
}
return(Res)
}


DescPart=function(Part, Names){
Freq=data.frame(table(Part))
Dim=dim(Freq)
NbMax=max(table(Part))
DescCluster=data.frame(matrix(0,Dim[1],NbMax))
for(i in 1:Dim[1]){
  VAR=Names[Part==i]
  for(j in 1:Freq$Freq[i]){
    DescCluster[i,j]=VAR[j]
  }
}
Freq=Freq$Freq
Cluster=paste(rep("Cl"),1:Dim[1],sep="")
DescCluster=cbind(Cluster,DescCluster,Freq)
return(DescCluster)
}

col0 = colorRampPalette(c('white', 'cyan', '#007FFF', 'blue','#00007F'))

col1 = colorRampPalette(c('#7F0000', 'red', '#FF7F00', 'yellow', 'white','cyan', '#007FFF', 'blue','#00007F'))

col2 = colorRampPalette(c('#67001F', '#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#FFFFFF', '#D1E5F0', '#92C5DE','#4393C3', '#2166AC', '#053061'))

col3 = colorRampPalette(c('red', 'white', 'blue'))

col4 = colorRampPalette(c('#7F0000', 'red', '#FF7F00', 'yellow', '#7FFF7F','cyan', '#007FFF', 'blue', '#00007F'))

wb = c('white', 'black')

par(ask = TRUE)