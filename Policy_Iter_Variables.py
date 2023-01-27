import numpy as np

def setup(m):
    if m == 1:
        #Big Model
        Probabilities = np.zeros((2, 154,154))
        Probabilities[0][0][0]=1/4
        Probabilities[0][0][1]=1/4
        Probabilities[0][0][2]=2/4
        Probabilities[0][1][0]=1/8
        Probabilities[0][1][1]=1/8
        Probabilities[0][1][5]=1/4
        Probabilities[0][1][6]=2/4
        Probabilities[0][2][0]=1/20
        Probabilities[0][2][2]=1/10
        Probabilities[0][2][9]=1/2
        Probabilities[0][2][10]=7/20
        Probabilities[0][3][3]=2/5
        Probabilities[0][3][13]=1/5
        Probabilities[0][3][14]=2/5
        Probabilities[0][4][4]=1/6
        Probabilities[0][4][17]=1/3
        Probabilities[0][4][18]=1/2
        Probabilities[0][5][1]=1/4
        Probabilities[0][5][5]=1/8
        Probabilities[0][5][21]=1/4
        Probabilities[0][5][22]=3/8
        Probabilities[0][6][1]=1/8
        Probabilities[0][6][6]=1/8
        Probabilities[0][6][25]=1/4
        Probabilities[0][6][26]=2/4
        Probabilities[0][7][7]=2/5
        Probabilities[0][7][29]=1/5
        Probabilities[0][7][30]=2/5
        Probabilities[0][8][8]=1/6
        Probabilities[0][8][33]=1/6
        Probabilities[0][8][34]=2/3
        Probabilities[0][9][2]=1/8
        Probabilities[0][9][9]=1/8
        Probabilities[0][9][37]=3/8
        Probabilities[0][9][38]=3/8
        Probabilities[0][10][2]=1/20
        Probabilities[0][10][10]=1/10
        Probabilities[0][10][41]=1/2
        Probabilities[0][10][42]=7/20
        Probabilities[0][11][11]=1/4
        Probabilities[0][11][45]=1/4
        Probabilities[0][11][46]=1/4
        Probabilities[0][12][12]=1/5
        Probabilities[0][12][49]=1/5
        Probabilities[0][12][50]=3/5
        Probabilities[0][13][3]=1/8
        Probabilities[0][13][13]=2/8
        Probabilities[0][13][53]=3/8
        Probabilities[0][13][54]=2/8
        Probabilities[0][14][3]=1/20
        Probabilities[0][14][14]=1/20
        Probabilities[0][14][57]=4/10
        Probabilities[0][14][58]=1/2
        Probabilities[0][15][15]=2/5
        Probabilities[0][15][61]=1/5
        Probabilities[0][15][62]=2/5
        Probabilities[0][16][16]=1/4
        Probabilities[0][16][65]=1/4
        Probabilities[0][16][66]=2/4
        Probabilities[0][17][4]=1/20
        Probabilities[0][17][17]=1/10
        Probabilities[0][17][69]=4/10
        Probabilities[0][17][70]=9/20
        Probabilities[0][18][4]=1/8
        Probabilities[0][18][18]=1/8
        Probabilities[0][18][73]=2/4
        Probabilities[0][18][74]=1/4
        Probabilities[0][19][19]=1/3
        Probabilities[0][19][77]=1/3
        Probabilities[0][19][78]=1/3
        Probabilities[0][20][20]=1/5
        Probabilities[0][20][81]=2/5
        Probabilities[0][20][82]=2/5
        Probabilities[0][21][5]=1/20
        Probabilities[0][21][21]=1/10
        Probabilities[0][21][85]=4/10
        Probabilities[0][21][86]=9/20
        Probabilities[0][22][5]=1/6
        Probabilities[0][22][22]=1/3
        Probabilities[0][22][85]=1/6
        Probabilities[0][22][86]=1/3
        Probabilities[0][23][23]=1/4
        Probabilities[0][23][87]=3/8
        Probabilities[0][23][88]=3/8
        Probabilities[0][24][24]=2/5
        Probabilities[0][24][87]=1/5
        Probabilities[0][24][88]=2/5
        Probabilities[0][25][6]=1/20
        Probabilities[0][25][25]=1/5
        Probabilities[0][25][89]=9/20
        Probabilities[0][25][90]=8/20
        Probabilities[0][26][6]=1/8
        Probabilities[0][26][26]=1/4
        Probabilities[0][26][89]=3/8
        Probabilities[0][26][90]=3/8
        Probabilities[0][27][27]=1/4
        Probabilities[0][27][91]=1/4
        Probabilities[0][27][92]=1/2
        Probabilities[0][28][28]=1/3
        Probabilities[0][28][91]=1/6
        Probabilities[0][28][92]=1/2
        Probabilities[0][29][7]=1/10
        Probabilities[0][29][29]=1/5
        Probabilities[0][29][93]=4/10
        Probabilities[0][29][94]=3/10
        Probabilities[0][30][7]=1/3
        Probabilities[0][30][30]=1/9
        Probabilities[0][30][93]=1/3
        Probabilities[0][30][94]=2/9
        Probabilities[0][31][31]=1/2
        Probabilities[0][31][95]=1/4
        Probabilities[0][31][96]=1/4
        Probabilities[0][32][32]=1/8
        Probabilities[0][32][95]=3/8
        Probabilities[0][32][96]=1/2
        Probabilities[0][33][8]=1/20
        Probabilities[0][33][33]=1/5
        Probabilities[0][33][97]=2/5
        Probabilities[0][33][98]=7/20
        Probabilities[0][34][8]=1/8
        Probabilities[0][34][34]=1/4
        Probabilities[0][34][97]=1/4
        Probabilities[0][34][98]=3/8
        Probabilities[0][35][35]=3/5
        Probabilities[0][35][99]=1/5
        Probabilities[0][35][100]=1/5
        Probabilities[0][36][36]=1/10
        Probabilities[0][36][99]=4/10
        Probabilities[0][36][100]=1/2
        Probabilities[0][37][9]=1/20
        Probabilities[0][37][37]=1/20
        Probabilities[0][37][101]=1/2
        Probabilities[0][37][102]=2/5
        Probabilities[0][38][9]=1/100
        Probabilities[0][38][38]=52/100
        Probabilities[0][38][103]=7/100
        Probabilities[0][38][104]=4/10
        Probabilities[0][39][9]=1/10
        Probabilities[0][39][39]=1/10
        Probabilities[0][39][103]=4/10
        Probabilities[0][39][104]=4/10
        Probabilities[0][40][40]=1/4
        Probabilities[0][40][105]=1/4
        Probabilities[0][40][106]=1/2
        Probabilities[0][41][41]=1/5
        Probabilities[0][41][105]=1/5
        Probabilities[0][41][106]=3/5
        Probabilities[0][42][10]=1/8
        Probabilities[0][42][42]=1/8
        Probabilities[0][42][107]=3/8
        Probabilities[0][42][108]=3/8
        Probabilities[0][43][10]=1/6
        Probabilities[0][43][43]=1/6
        Probabilities[0][43][107]=1/6
        Probabilities[0][43][108]=1/2
        Probabilities[0][44][44]=1/10
        Probabilities[0][44][109]=1/2
        Probabilities[0][44][110]=2/5
        Probabilities[0][45][45]=1/4
        Probabilities[0][45][109]=1/2
        Probabilities[0][45][110]=1/4
        Probabilities[0][46][11]=1/20
        Probabilities[0][46][46]=1/10
        Probabilities[0][46][111]=4/10
        Probabilities[0][46][112]=9/20
        Probabilities[0][47][11]=1/8
        Probabilities[0][47][47]=1/4
        Probabilities[0][47][111]=3/8
        Probabilities[0][47][112]=1/4
        Probabilities[0][48][48]=1/3
        Probabilities[0][48][113]=1/3
        Probabilities[0][48][114]=1/3
        Probabilities[0][49][49]=1/5
        Probabilities[0][49][113]=2/5
        Probabilities[0][49][114]=2/5
        Probabilities[0][50][12]=1/10
        Probabilities[0][50][50]=2/10
        Probabilities[0][50][115]=3/10
        Probabilities[0][50][116]=4/10
        Probabilities[0][51][12]=1/20
        Probabilities[0][51][51]=1/20
        Probabilities[0][51][115]=9/20
        Probabilities[0][51][116]=9/20
        Probabilities[0][52][52]=1/5
        Probabilities[0][52][117]=3/5
        Probabilities[0][52][118]=1/5
        Probabilities[0][53][53]=1/4
        Probabilities[0][53][119]=1/4
        Probabilities[0][53][120]=1/2
        Probabilities[0][54][13]=1/8
        Probabilities[0][54][54]=1/16
        Probabilities[0][54][121]=6/16
        Probabilities[0][54][122]=7/16
        Probabilities[0][55][13]=1/10
        Probabilities[0][55][55]=1/5
        Probabilities[0][55][121]=2/5
        Probabilities[0][55][122]=3/10
        Probabilities[0][56][56]=2/5
        Probabilities[0][56][123]=1/5
        Probabilities[0][56][124]=2/5
        Probabilities[0][57][57]=1/4
        Probabilities[0][57][123]=1/2
        Probabilities[0][57][124]=1/4
        Probabilities[0][58][14]=1/7
        Probabilities[0][58][58]=1/7
        Probabilities[0][58][125]=2/7
        Probabilities[0][58][126]=3/7
        Probabilities[0][59][14]=1/10
        Probabilities[0][59][59]=1/10
        Probabilities[0][59][125]=2/10
        Probabilities[0][59][126]=6/10
        Probabilities[0][60][60]=1/5
        Probabilities[0][60][127]=2/5
        Probabilities[0][60][128]=2/5
        Probabilities[0][61][61]=1/4
        Probabilities[0][61][127]=1/4
        Probabilities[0][61][128]=1/2
        Probabilities[0][62][15]=1/20
        Probabilities[0][62][62]=1/20
        Probabilities[0][62][129]=7/20
        Probabilities[0][62][130]=11/20
        Probabilities[0][63][15]=1/15
        Probabilities[0][63][63]=4/15
        Probabilities[0][63][129]=1/3
        Probabilities[0][63][130]=1/3
        Probabilities[0][64][64]=3/16
        Probabilities[0][64][131]=6/16
        Probabilities[0][64][132]=7/16
        Probabilities[0][65][65]=2/3
        Probabilities[0][65][131]=1/6
        Probabilities[0][65][132]=1/6
        Probabilities[0][66][16]=1/4
        Probabilities[0][66][66]=1/20
        Probabilities[0][66][133]=7/20
        Probabilities[0][66][134]=7/20
        Probabilities[0][67][16]=1/9
        Probabilities[0][67][67]=1/9
        Probabilities[0][67][133]=1/3
        Probabilities[0][67][134]=4/9
        Probabilities[0][68][68]=1/4
        Probabilities[0][68][135]=1/2
        Probabilities[0][68][136]=1/4
        Probabilities[0][69][69]=42/100
        Probabilities[0][69][135]=18/100
        Probabilities[0][69][136]=4/10
        Probabilities[0][70][17]=1/20
        Probabilities[0][70][70]=3/20
        Probabilities[0][70][137]=4/10
        Probabilities[0][70][138]=4/10
        Probabilities[0][71][17]=1/5
        Probabilities[0][71][71]=1/5
        Probabilities[0][71][137]=2/5
        Probabilities[0][71][138]=1/5
        Probabilities[0][72][72]=1/4
        Probabilities[0][72][139]=1/4
        Probabilities[0][72][140]=1/2
        Probabilities[0][73][73]=2/5
        Probabilities[0][73][139]=1/5
        Probabilities[0][73][140]=2/5
        Probabilities[0][74][18]=1/20
        Probabilities[0][74][74]=1/10
        Probabilities[0][74][141]=1/2
        Probabilities[0][74][142]=7/20
        Probabilities[0][75][18]=1/8
        Probabilities[0][75][75]=1/8
        Probabilities[0][75][141]=1/4
        Probabilities[0][75][142]=1/2
        Probabilities[0][76][76]=1/2
        Probabilities[0][76][143]=1/3
        Probabilities[0][76][144]=1/6
        Probabilities[0][77][77]=1/3
        Probabilities[0][77][143]=1/3
        Probabilities[0][77][144]=1/3
        Probabilities[0][78][19]=1/9
        Probabilities[0][78][78]=1/9
        Probabilities[0][78][145]=4/9
        Probabilities[0][78][146]=1/3
        Probabilities[0][79][19]=1/10
        Probabilities[0][79][79]=1/10
        Probabilities[0][79][145]=1/10
        Probabilities[0][79][146]=7/10
        Probabilities[0][80][80]=1/5
        Probabilities[0][80][147]=2/5
        Probabilities[0][80][148]=2/5
        Probabilities[0][81][81]=1/4
        Probabilities[0][81][147]=1/4
        Probabilities[0][81][148]=1/2
        Probabilities[0][82][20]=1/6
        Probabilities[0][82][82]=1/3
        Probabilities[0][82][149]=1/3
        Probabilities[0][82][150]=1/6
        Probabilities[0][83][20]=1/20
        Probabilities[0][83][83]=1/20
        Probabilities[0][83][149]=11/20
        Probabilities[0][83][150]=7/20
        Probabilities[0][84][84]=9/10
        Probabilities[0][84][151]=1/20
        Probabilities[0][84][152]=1/20
        Probabilities[0][85][21]=1/10
        Probabilities[0][85][85]=1/10
        Probabilities[0][85][153]=4/5
        Probabilities[0][86][22]=1/20
        Probabilities[0][86][86]=3/20
        Probabilities[0][86][153]=4/5
        Probabilities[0][87][23]=1/5
        Probabilities[0][87][87]=2/5
        Probabilities[0][87][153]=2/5
        Probabilities[0][88][24]=1/8
        Probabilities[0][88][88]=1/4
        Probabilities[0][88][153]=5/8
        Probabilities[0][89][25]=1/3
        Probabilities[0][89][89]=1/6
        Probabilities[0][89][153]=1/2
        Probabilities[0][90][26]=1/10
        Probabilities[0][90][90]=2/10
        Probabilities[0][90][153]=7/10
        Probabilities[0][91][27]=1/7
        Probabilities[0][91][91]=4/7
        Probabilities[0][91][153]=2/7
        Probabilities[0][92][28]=1/4
        Probabilities[0][92][92]=1/4
        Probabilities[0][92][153]=1/2
        Probabilities[0][93][29]=1/3
        Probabilities[0][93][93]=1/3
        Probabilities[0][93][153]=1/3
        Probabilities[0][94][30]=1/15
        Probabilities[0][94][94]=2/15
        Probabilities[0][94][153]=12/15
        Probabilities[0][95][31]=1/9
        Probabilities[0][95][95]=1/9
        Probabilities[0][95][153]=7/9
        Probabilities[0][96][32]=1/4
        Probabilities[0][96][96]=1/2
        Probabilities[0][96][153]=1/4
        Probabilities[0][97][33]=1/6
        Probabilities[0][97][97]=1/3
        Probabilities[0][97][153]=1/2
        Probabilities[0][98][33]=1/7
        Probabilities[0][98][98]=2/7
        Probabilities[0][98][153]=4/7
        Probabilities[0][99][34]=1/8
        Probabilities[0][99][99]=3/8
        Probabilities[0][99][153]=1/2
        Probabilities[0][100][35]=1/3
        Probabilities[0][100][100]=1/3
        Probabilities[0][100][153]=1/3
        Probabilities[0][101][36]=1/5
        Probabilities[0][101][101]=2/5
        Probabilities[0][101][153]=2/5
        Probabilities[0][102][37]=1/10
        Probabilities[0][102][102]=1/10
        Probabilities[0][102][153]=4/5
        Probabilities[0][103][38]=1/6
        Probabilities[0][103][103]=1/6
        Probabilities[0][103][153]=2/3
        Probabilities[0][104][39]=1/20
        Probabilities[0][104][104]=1/20
        Probabilities[0][104][153]=9/10
        Probabilities[0][105][40]=1/3
        Probabilities[0][105][105]=1/3
        Probabilities[0][105][153]=1/3
        Probabilities[0][106][41]=1/7
        Probabilities[0][106][106]=1/7
        Probabilities[0][106][153]=5/7
        Probabilities[0][107][42]=1/5
        Probabilities[0][107][107]=1/5
        Probabilities[0][107][153]=3/5
        Probabilities[0][108][43]=1/8
        Probabilities[0][108][108]=1/4
        Probabilities[0][108][153]=5/8
        Probabilities[0][109][44]=1/11
        Probabilities[0][109][109]=2/11
        Probabilities[0][109][153]=8/11
        Probabilities[0][110][45]=1/6
        Probabilities[0][110][110]=1/2
        Probabilities[0][110][153]=1/3
        Probabilities[0][111][46]=1/12
        Probabilities[0][111][111]=1/6
        Probabilities[0][111][153]=3/4
        Probabilities[0][112][47]=1/9
        Probabilities[0][112][112]=1/3
        Probabilities[0][112][153]=5/9
        Probabilities[0][113][48]=1/4
        Probabilities[0][113][113]=1/4
        Probabilities[0][113][153]=1/2
        Probabilities[0][114][49]=1/10
        Probabilities[0][114][114]=1/5
        Probabilities[0][114][153]=7/10
        Probabilities[0][115][50]=4/100
        Probabilities[0][115][115]=12/100
        Probabilities[0][115][153]=84/100
        Probabilities[0][116][51]=2/7
        Probabilities[0][116][116]=1/7
        Probabilities[0][116][153]=4/7
        Probabilities[0][117][52]=1/8
        Probabilities[0][117][117]=1/8
        Probabilities[0][117][153]=3/4
        Probabilities[0][118][53]=1/6
        Probabilities[0][118][118]=1/3
        Probabilities[0][118][153]=1/2
        Probabilities[0][119][54]=1/12
        Probabilities[0][119][119]=1/12
        Probabilities[0][119][153]=5/6
        Probabilities[0][120][55]=1/5
        Probabilities[0][120][120]=1/5
        Probabilities[0][120][153]=3/5
        Probabilities[0][121][56]=1/9
        Probabilities[0][121][121]=1/9
        Probabilities[0][121][153]=7/9
        Probabilities[0][122][57]=1/4
        Probabilities[0][122][122]=1/4
        Probabilities[0][122][153]=1/2
        Probabilities[0][123][58]=1/10
        Probabilities[0][123][123]=1/5
        Probabilities[0][123][153]=7/10
        Probabilities[0][124][59]=1/5
        Probabilities[0][124][124]=3/5
        Probabilities[0][124][153]=1/5
        Probabilities[0][125][60]=1/20
        Probabilities[0][125][125]=1/5
        Probabilities[0][125][153]=3/4
        Probabilities[0][126][61]=1/8
        Probabilities[0][126][126]=1/8
        Probabilities[0][126][153]=3/4
        Probabilities[0][127][62]=1/12
        Probabilities[0][127][127]=4/12
        Probabilities[0][127][153]=7/12
        Probabilities[0][128][63]=1/10
        Probabilities[0][128][128]=3/10
        Probabilities[0][128][153]=3/5
        Probabilities[0][129][64]=1/7
        Probabilities[0][129][129]=2/7
        Probabilities[0][129][153]=4/7
        Probabilities[0][130][65]=1/5
        Probabilities[0][130][130]=1/5
        Probabilities[0][130][153]=3/5
        Probabilities[0][131][66]=1/3
        Probabilities[0][131][131]=1/3
        Probabilities[0][131][153]=1/3
        Probabilities[0][132][67]=1/2
        Probabilities[0][132][132]=1/3
        Probabilities[0][132][153]=1/6
        Probabilities[0][133][68]=1/20
        Probabilities[0][133][133]=1/4
        Probabilities[0][133][153]=7/10
        Probabilities[0][134][69]=1/15
        Probabilities[0][134][134]=1/3
        Probabilities[0][134][153]=9/15
        Probabilities[0][135][70]=1/6
        Probabilities[0][135][135]=1/6
        Probabilities[0][135][153]=2/3
        Probabilities[0][136][71]=1/7
        Probabilities[0][136][136]=2/7
        Probabilities[0][136][153]=4/7
        Probabilities[0][137][72]=1/9
        Probabilities[0][137][137]=1/3
        Probabilities[0][137][153]=5/9
        Probabilities[0][138][73]=1/5
        Probabilities[0][138][138]=2/5
        Probabilities[0][138][153]=2/5
        Probabilities[0][139][74]=1/6
        Probabilities[0][139][139]=1/3
        Probabilities[0][139][153]=1/2
        Probabilities[0][140][75]=1/8
        Probabilities[0][140][140]=1/4
        Probabilities[0][140][153]=5/8
        Probabilities[0][141][76]=1/10
        Probabilities[0][141][141]=1/5
        Probabilities[0][141][153]=7/10
        Probabilities[0][142][77]=1/20
        Probabilities[0][142][142]=3/10
        Probabilities[0][142][153]=13/20
        Probabilities[0][143][78]=1/6
        Probabilities[0][143][143]=1/6
        Probabilities[0][143][153]=2/3
        Probabilities[0][144][79]=1/7
        Probabilities[0][144][144]=4/7
        Probabilities[0][144][153]=2/7
        Probabilities[0][145][80]=1/8
        Probabilities[0][145][145]=3/8
        Probabilities[0][145][153]=1/2
        Probabilities[0][146][81]=1/10
        Probabilities[0][146][146]=1/10
        Probabilities[0][146][153]=4/5
        Probabilities[0][147][82]=1/5
        Probabilities[0][147][147]=1/5
        Probabilities[0][147][153]=3/5
        Probabilities[0][148][83]=1/7
        Probabilities[0][148][148]=1/7
        Probabilities[0][148][153]=5/7
        Probabilities[0][149][84]=1/3
        Probabilities[0][149][149]=1/3
        Probabilities[0][149][153]=1/3
        Probabilities[0][150][85]=1/9
        Probabilities[0][150][150]=1/6
        Probabilities[0][150][153]=2/3
        Probabilities[0][151][86]=1/10
        Probabilities[0][151][151]=1/5
        Probabilities[0][151][153]=7/10
        Probabilities[0][152][87]=1/5
        Probabilities[0][152][152]=1/5
        Probabilities[0][152][153]=3/5
        Probabilities[0][153][153]=1
        #new action
        Probabilities[1][0][0]=1/2
        Probabilities[1][0][3]=1/4
        Probabilities[1][0][4]=1/4
        Probabilities[1][1][1]=1/2
        Probabilities[1][1][7]=1/4
        Probabilities[1][1][8]=1/4
        Probabilities[1][2][2]=7/20
        Probabilities[1][2][11]=1/2
        Probabilities[1][2][12]=3/20
        Probabilities[1][3][3]=2/5
        Probabilities[1][3][15]=1/5
        Probabilities[1][3][16]=2/5
        Probabilities[1][4][4]=1/2
        Probabilities[0][4][19]=1/3
        Probabilities[1][4][20]=1/6
        Probabilities[1][5][5]=3/8
        Probabilities[1][5][23]=1/4
        Probabilities[1][5][24]=1/8
        Probabilities[1][6][6]=1/2
        Probabilities[1][6][27]=1/4
        Probabilities[1][6][28]=1/4
        Probabilities[1][7][7]=2/5
        Probabilities[1][7][31]=1/5
        Probabilities[1][7][32]=2/5
        Probabilities[1][8][8]=2/3
        Probabilities[1][8][35]=1/6
        Probabilities[1][8][36]=1/6
        Probabilities[1][9][9]=3/8
        Probabilities[1][9][39]=3/8
        Probabilities[1][9][40]=2/8
        Probabilities[1][10][10]=7/20
        Probabilities[1][10][43]=1/2
        Probabilities[1][10][44]=3/20
        Probabilities[1][11][11]=1/3
        Probabilities[1][11][47]=1/3
        Probabilities[1][11][48]=1/3
        Probabilities[1][12][12]=3/5
        Probabilities[1][12][51]=1/5
        Probabilities[1][12][52]=1/5
        Probabilities[1][13][13]=2/8
        Probabilities[1][13][55]=4/8
        Probabilities[1][13][56]=2/8
        Probabilities[1][14][14]=1/2
        Probabilities[1][14][59]=4/10
        Probabilities[1][14][60]=1/10
        Probabilities[1][15][15]=2/5
        Probabilities[1][15][63]=1/5
        Probabilities[1][15][64]=2/5
        Probabilities[1][16][16]=2/4
        Probabilities[1][16][67]=1/4
        Probabilities[1][16][68]=1/4
        Probabilities[1][17][17]=9/20
        Probabilities[1][17][71]=4/10
        Probabilities[1][17][72]=3/20
        Probabilities[1][18][18]=1/4
        Probabilities[1][18][75]=2/4
        Probabilities[1][18][76]=1/4
        Probabilities[1][19][19]=1/6
        Probabilities[1][19][79]=1/6
        Probabilities[1][19][80]=2/3
        Probabilities[1][20][20]=2/5
        Probabilities[1][20][83]=2/5
        Probabilities[1][20][84]=1/5
        Probabilities[1][21][21]=4/10
        Probabilities[1][21][85]=3/20
        Probabilities[1][21][86]=9/20
        Probabilities[1][22][22]=1/3
        Probabilities[1][22][85]=1/3
        Probabilities[1][22][86]=1/3
        Probabilities[1][23][23]=3/8
        Probabilities[1][23][87]=1/4
        Probabilities[1][23][88]=3/8
        Probabilities[1][24][24]=1/5
        Probabilities[1][24][87]=2/5
        Probabilities[1][24][88]=2/5
        Probabilities[1][25][25]=9/20
        Probabilities[1][25][89]=1/5
        Probabilities[1][25][90]=7/20
        Probabilities[1][26][26]=3/8
        Probabilities[1][26][89]=1/4
        Probabilities[1][26][90]=3/8
        Probabilities[1][27][27]=1/2
        Probabilities[1][27][91]=1/4
        Probabilities[1][27][92]=1/4
        Probabilities[1][28][28]=1/6
        Probabilities[1][28][91]=1/3
        Probabilities[1][28][92]=1/2
        Probabilities[1][29][29]=4/10
        Probabilities[1][29][93]=4/10
        Probabilities[1][29][94]=2/10
        Probabilities[1][30][30]=2/3
        Probabilities[1][30][93]=1/9
        Probabilities[1][30][94]=2/9
        Probabilities[1][31][31]=1/4
        Probabilities[1][31][95]=1/2
        Probabilities[1][31][96]=1/4
        Probabilities[1][32][32]=3/8
        Probabilities[1][32][95]=1/8
        Probabilities[1][32][96]=1/2
        Probabilities[1][33][33]=2/5
        Probabilities[1][33][97]=1/4
        Probabilities[1][33][98]=7/20
        Probabilities[1][34][34]=1/4
        Probabilities[1][34][97]=1/4
        Probabilities[1][34][98]=1/2
        Probabilities[1][35][35]=1/5
        Probabilities[1][35][99]=3/5
        Probabilities[1][35][100]=1/5
        Probabilities[1][36][36]=4/10
        Probabilities[1][36][99]=1/10
        Probabilities[1][36][100]=1/2
        Probabilities[1][37][37]=1/2
        Probabilities[1][37][101]=1/10
        Probabilities[1][37][102]=2/5
        Probabilities[1][38][38]=7/100
        Probabilities[1][38][103]=53/100
        Probabilities[1][38][104]=4/10
        Probabilities[1][39][39]=4/10
        Probabilities[1][39][103]=2/10
        Probabilities[1][39][104]=4/10
        Probabilities[1][40][40]=1/4
        Probabilities[1][40][105]=1/4
        Probabilities[1][40][106]=1/2
        Probabilities[1][41][41]=3/5
        Probabilities[1][41][105]=1/5
        Probabilities[1][41][106]=1/5
        Probabilities[1][42][42]=3/8
        Probabilities[1][42][107]=3/8
        Probabilities[1][42][108]=1/4
        Probabilities[1][43][43]=1/2
        Probabilities[1][43][107]=1/6
        Probabilities[1][43][108]=1/3
        Probabilities[1][44][44]=1/2
        Probabilities[1][44][109]=1/10
        Probabilities[1][44][110]=2/5
        Probabilities[1][45][45]=1/2
        Probabilities[1][45][109]=1/4
        Probabilities[1][45][110]=1/4
        Probabilities[1][46][46]=4/10
        Probabilities[1][46][111]=1/10
        Probabilities[1][46][112]=1/2
        Probabilities[1][47][47]=3/8
        Probabilities[1][47][111]=3/8
        Probabilities[1][47][112]=1/4
        Probabilities[1][48][48]=1/3
        Probabilities[1][48][113]=1/3
        Probabilities[1][48][114]=1/3
        Probabilities[1][49][49]=2/5
        Probabilities[1][49][113]=2/5
        Probabilities[1][49][114]=1/5
        Probabilities[1][50][50]=4/10
        Probabilities[1][50][115]=3/10
        Probabilities[1][50][116]=3/10
        Probabilities[1][51][51]=9/20
        Probabilities[1][51][115]=9/20
        Probabilities[1][51][116]=2/20
        Probabilities[1][52][52]=1/5
        Probabilities[1][52][117]=3/5
        Probabilities[1][52][118]=1/5
        Probabilities[1][53][53]=1/2
        Probabilities[1][53][119]=1/4
        Probabilities[1][53][120]=1/4
        Probabilities[1][54][54]=7/16
        Probabilities[1][54][121]=6/16
        Probabilities[1][54][122]=3/16
        Probabilities[1][55][55]=3/10
        Probabilities[1][55][121]=2/5
        Probabilities[1][55][122]=3/10
        Probabilities[1][56][56]=2/5
        Probabilities[1][56][123]=1/5
        Probabilities[1][56][124]=2/5
        Probabilities[1][57][57]=1/4
        Probabilities[1][57][123]=1/2
        Probabilities[1][57][124]=1/4
        Probabilities[1][58][58]=3/7
        Probabilities[1][58][125]=2/7
        Probabilities[1][58][126]=2/7
        Probabilities[1][59][59]=6/10
        Probabilities[1][59][125]=2/10
        Probabilities[1][59][126]=2/10
        Probabilities[1][60][60]=2/5
        Probabilities[1][60][127]=2/5
        Probabilities[1][60][128]=1/5
        Probabilities[1][61][61]=1/2
        Probabilities[1][61][127]=1/4
        Probabilities[1][61][128]=1/4
        Probabilities[1][62][62]=11/20
        Probabilities[1][62][129]=7/20
        Probabilities[1][62][130]=2/20
        Probabilities[1][63][63]=1/3
        Probabilities[1][63][129]=1/3
        Probabilities[1][63][130]=1/3
        Probabilities[1][64][64]=7/16
        Probabilities[1][64][131]=6/16
        Probabilities[1][64][132]=3/16
        Probabilities[1][65][65]=1/6
        Probabilities[1][65][131]=1/6
        Probabilities[1][65][132]=2/3
        Probabilities[1][66][66]=7/20
        Probabilities[1][66][133]=7/20
        Probabilities[1][66][134]=6/20
        Probabilities[1][67][67]=4/9
        Probabilities[1][67][133]=1/3
        Probabilities[1][67][134]=2/9
        Probabilities[1][68][68]=1/4
        Probabilities[1][68][135]=1/2
        Probabilities[1][68][136]=1/4
        Probabilities[1][69][69]=4/10
        Probabilities[1][69][135]=18/100
        Probabilities[1][69][136]=42/100
        Probabilities[1][70][70]=4/10
        Probabilities[1][70][137]=4/10
        Probabilities[1][70][138]=2/10
        Probabilities[1][71][71]=1/5
        Probabilities[1][71][137]=2/5
        Probabilities[1][71][138]=1/5
        Probabilities[1][72][72]=1/2
        Probabilities[1][72][139]=1/4
        Probabilities[1][72][140]=1/4
        Probabilities[1][73][73]=2/5
        Probabilities[1][73][139]=1/5
        Probabilities[1][73][140]=2/5
        Probabilities[1][74][74]=7/20
        Probabilities[1][74][141]=1/2
        Probabilities[1][74][142]=3/20
        Probabilities[1][75][75]=1/2
        Probabilities[1][75][141]=1/4
        Probabilities[1][75][142]=1/4
        Probabilities[1][76][76]=1/6
        Probabilities[1][76][143]=1/3
        Probabilities[1][76][144]=1/2
        Probabilities[1][77][77]=1/3
        Probabilities[1][77][143]=1/3
        Probabilities[1][77][144]=1/3
        Probabilities[1][78][78]=1/3
        Probabilities[1][78][145]=4/9
        Probabilities[1][78][146]=2/9
        Probabilities[1][79][79]=7/10
        Probabilities[1][79][145]=2/10
        Probabilities[1][79][146]=1/10
        Probabilities[1][80][80]=2/5
        Probabilities[1][80][147]=2/5
        Probabilities[1][80][148]=1/5
        Probabilities[1][81][81]=1/4
        Probabilities[1][81][147]=1/4
        Probabilities[1][81][148]=1/2
        Probabilities[1][82][82]=1/6
        Probabilities[1][82][149]=1/3
        Probabilities[1][82][150]=1/2
        Probabilities[1][83][83]=11/20
        Probabilities[1][83][149]=2/20
        Probabilities[1][83][150]=7/20
        Probabilities[1][84][84]=1/20
        Probabilities[1][84][151]=9/10
        Probabilities[1][84][152]=1/20
        Probabilities[1][85][85]=4/5
        Probabilities[1][85][153]=1/5
        Probabilities[1][86][86]=1/5
        Probabilities[1][86][153]=4/5
        Probabilities[1][87][87]=2/5
        Probabilities[1][87][153]=3/5
        Probabilities[1][88][88]=1/4
        Probabilities[1][88][153]=3/4
        Probabilities[1][89][89]=1/20
        Probabilities[1][89][153]=19/20
        Probabilities[1][90][90]=8/10
        Probabilities[1][90][153]=2/10
        Probabilities[1][91][91]=4/7
        Probabilities[1][91][153]=3/7
        Probabilities[1][92][92]=1/2
        Probabilities[1][92][153]=1/2
        Probabilities[1][93][93]=4/9
        Probabilities[1][93][153]=5/9
        Probabilities[1][94][94]=12/15
        Probabilities[1][94][153]=3/15
        Probabilities[1][95][95]=8/9
        Probabilities[1][95][153]=1/9
        Probabilities[1][96][96]=1/2
        Probabilities[1][96][153]=1/2
        Probabilities[1][97][97]=1/3
        Probabilities[1][97][153]=2/3
        Probabilities[1][98][98]=2/7
        Probabilities[1][98][153]=5/7
        Probabilities[1][99][99]=3/8
        Probabilities[1][99][153]=5/8
        Probabilities[1][100][100]=1/10
        Probabilities[1][100][153]=9/10
        Probabilities[1][101][101]=1/4
        Probabilities[1][101][153]=3/4
        Probabilities[1][102][102]=1/10
        Probabilities[1][102][153]=9/10
        Probabilities[1][103][103]=2/3
        Probabilities[1][103][153]=1/3
        Probabilities[1][104][104]=3/10
        Probabilities[1][104][153]=7/10
        Probabilities[1][105][105]=1/2
        Probabilities[1][105][153]=1/2
        Probabilities[1][106][106]=3/7
        Probabilities[1][106][153]=4/7
        Probabilities[1][107][107]=2/5
        Probabilities[1][107][153]=3/5
        Probabilities[1][108][108]=1/2
        Probabilities[1][108][153]=1/2
        Probabilities[1][109][109]=8/11
        Probabilities[1][109][153]=3/11
        Probabilities[1][110][110]=2/3
        Probabilities[1][110][153]=1/3
        Probabilities[1][111][111]=3/4
        Probabilities[1][111][153]=1/4
        Probabilities[1][112][112]=5/9
        Probabilities[1][112][153]=4/9
        Probabilities[1][113][113]=1/4
        Probabilities[1][113][153]=3/4
        Probabilities[1][114][114]=1/5
        Probabilities[1][114][153]=4/5
        Probabilities[1][115][115]=86/100
        Probabilities[1][115][153]=14/100
        Probabilities[1][116][116]=1/7
        Probabilities[1][116][153]=6/7
        Probabilities[1][117][117]=1/8
        Probabilities[1][117][153]=7/8
        Probabilities[1][118][118]=1/2
        Probabilities[1][118][153]=1/2
        Probabilities[1][119][119]=1/12
        Probabilities[1][119][153]=11/12
        Probabilities[1][120][120]=4/5
        Probabilities[1][120][153]=1/5
        Probabilities[1][121][121]=2/9
        Probabilities[1][121][153]=7/9
        Probabilities[1][122][122]=4/30
        Probabilities[1][122][153]=26/30
        Probabilities[1][123][123]=1/5
        Probabilities[1][123][153]=4/5
        Probabilities[1][124][124]=3/5
        Probabilities[1][124][153]=2/5
        Probabilities[1][125][125]=2/5
        Probabilities[1][125][153]=3/5
        Probabilities[1][126][126]=1/4
        Probabilities[1][126][153]=3/4
        Probabilities[1][127][127]=5/12
        Probabilities[1][127][153]=7/12
        Probabilities[1][128][128]=3/5
        Probabilities[1][128][153]=2/5
        Probabilities[1][129][129]=5/7
        Probabilities[1][129][153]=2/7
        Probabilities[1][130][130]=1/5
        Probabilities[1][130][153]=4/5
        Probabilities[1][131][131]=1/2
        Probabilities[1][131][153]=1/2
        Probabilities[1][132][132]=2/3
        Probabilities[1][132][153]=1/3
        Probabilities[1][133][133]=1/4
        Probabilities[1][133][153]=3/4
        Probabilities[1][134][134]=1/3
        Probabilities[1][134][153]=2/3
        Probabilities[1][135][135]=1/6
        Probabilities[1][135][153]=5/6
        Probabilities[1][136][136]=2/7
        Probabilities[1][136][153]=5/7
        Probabilities[1][137][137]=4/9
        Probabilities[1][137][153]=5/9
        Probabilities[1][138][138]=2/15
        Probabilities[1][138][153]=13/15
        Probabilities[1][139][139]=1/3
        Probabilities[1][139][153]=2/3
        Probabilities[1][140][140]=1/4
        Probabilities[1][140][153]=3/4
        Probabilities[1][141][141]=1/5
        Probabilities[1][141][153]=4/5
        Probabilities[1][142][142]=7/10
        Probabilities[1][142][153]=3/10
        Probabilities[1][143][143]=4/6
        Probabilities[1][143][153]=1/3
        Probabilities[1][144][144]=2/7
        Probabilities[1][144][153]=5/7
        Probabilities[1][145][145]=1/2
        Probabilities[1][145][153]=1/2
        Probabilities[1][146][146]=1/5
        Probabilities[1][146][153]=4/5
        Probabilities[1][147][147]=2/5
        Probabilities[1][147][153]=3/5
        Probabilities[1][148][148]=5/7
        Probabilities[1][148][153]=2/7
        Probabilities[1][149][149]=2/3
        Probabilities[1][149][153]=1/3
        Probabilities[1][150][150]=4/6
        Probabilities[1][150][153]=1/3
        Probabilities[1][151][151]=7/10
        Probabilities[1][151][153]=3/10
        Probabilities[1][152][152]=77/100
        Probabilities[1][152][153]=23/100
        Probabilities[1][153][153]=1


        cost = np.array([[ 67. , 97.],
                        [ 15. , 20.],
                        [ 94. , 29.],
                        [ 46. ,  6.],
                        [ 93. , 44.],
                        [ 61.,  68.],
                        [ 10. , 65.],
                        [ 46. , 33.],
                        [ 96. ,  7.],
                        [ 32. , 97.],
                        [ 14. , 99.],
                        [ 22. , 21.],
                        [ 75. , 68.],
                        [ 72. , 85.],
                        [ 44. ,  9.],
                        [ 12. , 18.],
                        [ 87. , 16.],
                        [ 31. , 84.],
                        [ 30. , 84.],
                        [ 52. , 51.],
                        [ 61. , 43.],
                        [ 15. , 55.],
                        [ 92. , 19.],
                        [ 72. , 25.],
                        [ 13. , 81.],
                        [ 40. ,  8.],
                        [ 52. ,  1.],
                        [ 68. , 79.],
                        [ 79. , 98.],
                        [ 51. , 55.],
                        [ 58. ,  8.],
                        [ 38. ,  1.],
                        [ 25. , 26.],
                        [ 59. , 44.],
                        [ 80. , 95.],
                        [  3. , 15.],
                        [ 89. ,  7.],
                        [ 16. , 94.],
                        [ 37. , 98.],
                        [ 53. , 64.],
                        [ 76. , 13.],
                        [ 28. , 75.],
                        [ 48. ,  3.],
                        [ 28. , 15.],
                        [ 30. , 76.],
                        [ 66. , 59.],
                        [ 35. , 84.],
                        [ 61. , 51.],
                        [ 76. , 43.],
                        [ 25. , 44.],
                        [  8. ,  5.],
                        [ 31. , 13.],
                        [  2. , 39.],
                        [ 11. , 90.],
                        [ 65. , 44.],
                        [ 13. , 52.],
                        [ 14. , 42.],
                        [ 83. , 48.],
                        [ 74. , 56.],
                        [ 31. , 52.],
                        [ 12. , 63.],
                        [ 86.  , 4.],
                        [ 90. , 55.],
                        [ 78. , 49.],
                        [ 73. , 95.],
                        [ 10. , 92.],
                        [ 35. , 45.],
                        [ 37. ,  8.],
                        [ 66. , 84.],
                        [ 29. ,  2.],
                        [ 88. , 71.],
                        [ 11. , 58.],
                        [ 20. , 63.],
                        [ 22. , 72.],
                        [ 58. , 42.],
                        [ 87. , 61.],
                        [ 74. , 82.],
                        [  9. , 57.],
                        [ 66. , 65.],
                        [ 82. , 24.],
                        [ 75. , 38.],
                        [ 96. , 62.],
                        [ 11. ,  2.],
                        [ 75. , 96.],
                        [ 23. , 81.],
                        [ 96. , 21.],
                        [  1. , 44.],
                        [ 68. , 10.],
                        [ 83. , 13.],
                        [ 31. , 45.],
                        [  1. , 17.],
                        [ 83. , 26.],
                        [ 76. , 97.],
                        [100. , 55.],
                        [ 58. , 19.],
                        [ 59. , 72.],
                        [ 55. , 89.],
                        [ 93. , 15.],
                        [ 64. , 12.],
                        [ 46. , 72.],
                        [  2. , 91.],
                        [ 58. , 83.],
                        [ 52. , 13.],
                        [ 90. , 77.],
                        [ 44. , 24.],
                        [ 41. ,  2.],
                        [ 82. , 47.],
                        [  6. , 19.],
                        [ 75. , 20.],
                        [ 77. , 63.],
                        [ 16. , 37.],
                        [ 18. , 23.],
                        [ 40. , 83.],
                        [ 69. , 38.],
                        [ 51. , 66.],
                        [ 59. ,  0.],
                        [ 40. , 87.],
                        [ 22. , 47.],
                        [ 95. , 54.],
                        [  3. , 78.],
                        [  3. , 60.],
                        [ 40. , 73.],
                        [ 17. , 78.],
                        [ 38. , 20.],
                        [ 82. , 41.],
                        [ 65.  , 9.],
                        [ 47. , 41.],
                        [ 66. , 91.],
                        [ 86. , 58.],
                        [ 92. ,  3.],
                        [ 93. , 34.],
                        [ 17. , 84.],
                        [ 97. , 19.],
                        [ 46. ,  7.],
                        [ 14. , 25.],
                        [ 46. , 45.],
                        [ 85. , 66.],
                        [ 12. , 61.],
                        [ 55. , 82.],
                        [ 61. , 79.],
                        [  1. , 33.],
                        [100. , 77.],
                        [  8. , 84.],
                        [ 74. , 17.],
                        [ 72. , 66.],
                        [ 64.,  49.],
                        [ 60.,  55.],
                        [ 21.,  27.],
                        [ 94.,  86.],
                        [ 67.,  7.],
                        [  3.,  90.],
                        [  8.,  70.],
                        [ 65.,  39.],
                        [ 61.,  36.]])
    if m == 2:
        #small Model for testing
        Probabilities = np.array([[[1/3, 1/3, 1/3],[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]],
                                 [[2/4, 1/4, 1/4], [2/4, 1/4, 1/4], [2/4, 1/4, 1/4]],
                                 [[1/2, 1/2, 0], [1/2, 1/2, 0], [1/2, 1/2, 0]]])

        cost = np.array([[1, 2, 3],
                        [2, 3, 1],
                        [3, 1, 2]])
    if m==3:
        #how to get to work example Note: States are in order [Home, Bike, Low Traffic, Medium Traffic, High Traffic, Train, Waiting Room, Work] and actions are in order: [take Bike, use Car, go to trainstation, wait, go home, drive]
        Probabilities = np.array([[[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]],
                                 [[0,0,1/6,2/3,1/6,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]],
                                 [[0,0,0,0,0,2/3,1/3,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]],
                                 [[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,2/3,1/3,0],[0,0,0,0,0,0,0,1]],
                                 [[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]],
                                 [[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1]]])

        cost = np.array([[0, 0, 5, 1e06, 1e06, 1e06],
                         [1e06, 1e06, 1e06, 1e06, 1e06, 45],
                         [1e06, 1e06, 1e06, 1e06, 1e06, 15],
                         [1e06, 1e06, 1e06, 1e06, 1e06, 30],
                         [1e06, 1e06, 1e06, 1e06, 1e06, 70],
                         [1e06, 1e06, 1e06, 1e06, 1e06, 25],
                         [1e06, 1e06, 1e06, 3, 5, 1e06],
                         [1e06, 1e06, 1e06, 1e06, 0, 1e06]])
    return Probabilities, cost
