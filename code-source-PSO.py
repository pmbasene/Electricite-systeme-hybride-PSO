#!/anaconda3/bin/python

import numpy as np 
import pandas as pd
import math
import datetime
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# from pylab import *
# from drawnow import drawnow, figure
# tmps1=time.clock()


def charge(Puissance_pv, Energie_Batterie, Energie_Batterie_max , Puissance_charge, temps_iterative , Energie_flexible, Energie_charge,uinv=0.9): 

	Pch[t]= P_pvout [t]- (charge_jr[t]//uinv)     # la puissance residuelle 
												 #pour eventuellement charger les batteries

	Ech[t] = Pch[t]

	if Ech[t] <= Ebmax - Eb[t]:       #on fixe quand est ce qu'on doit autoriser la charge des batteries

		Eb[t] = Eb[t-1]  + Ech[t] 		# la charge de la batterie à l'instant t+1 s
											#	i la condition elle est remplie
										       
		if Eb[t] > Ebmax: 

			Edump[t] = Ech[t] - (Ebmax - Eb[t])

			Eb[t] = Ebmax 
		else: 
			Edump[t] = 1            
	else: 
		Eb[t] = Ebmax 

		Edump[t]= Ech[t] - (Ebmax - Eb[t])

	return Ech,	Eb ,Edump
def decharge(Puissance_pv, Energie_Batterie, Energie_Batterie_max ,Puissance_decharge, temps_iterative, Puissance_generateur, Energie_Batterie_min, Energie_flexible,  diesel, time1 ,uinv=0.9):
 #11 arg obligatoires
 #j'ai enlevé Energie_charge car je vois son utilité pour le moment

    Pdch[t] =(charge_jr[t]/uinv) - P_pvout[t]   # la puissance deficitaire à compenser par l'appoint de GE
 
    Edch[t]= Pdch[t]  

    #Ebrest[t-1]=Eb[t-1] - Ebmin             # energie restante au niveau de la batterie
    
    if Eb[t-1] - Ebmin >= Edch[t]:
        Eb[t] = Eb[t-1] - Edch[t]

        time1[t]= 2 
        """si on observe bien les valeurs printed de Eb[t-1] - Ebmin et Edch
         on voit la condition if est toujours satisfaite dans ce cas de figure 
         donc on entrera pas dans la structure conditionnelle else. et par ricochet 
         la vateur Edump sera toujours egale a sa valeur declarative cad nulle"""

    else:
    	Eb[t] = (Eb[t-1] + (Png * uinv) + P_pvout[t]) - ((charge_jr[t]/uinv))

    	if Eb[t] > Ebmax:

    		Edump[t] = Eb[t] - Ebmax

    		Eb[t]   = Ebmax

    	elif Eb[t]< Ebmin:

    		Edump[t]=3

    		Eb[t]=Ebmin

    		diesel[t]=Png* uinv

    return Edch, Eb, time1, Edump, diesel
def economic(diesel, charge_jr, Fg, Pdies_nom1  ,max_deficit_charge_reel, p_npv,  param=1):
	"""cwh=== max_deficit_charge_reel    energie reservée pour un autonomie de 2  
		# Rappelons  que le systeme est composé de : pv , batt. regulateur, inv, GE

		# investement du systeme global ++++ COUT++++
			# les couts sont evalués au kW cad euros/kW.et reprensentent les couts initiaux
	 DECLARATTION DES CONSTANTES	"""
	PV_C =1400        # en Euros/kW      # Prix d'achat des composantes  !!!! les prix ne sont pas justes
	PV_reg_C=480
	INV_C =600
	DSL_C = 411
	BAT_C =192


	REAL_INTERST = 12

		# duree de vie des differents composants / an

	PV_LF = 20  		# en Année
	PV_reg_LF= 5		# en Année
	INV_LF = 10			# en Année
	DSL_LF = 2400		# en kW
	BAT_LF = 8			# en Année
	PRJ_LF = 20			# en Année


	# OM = operating (fonctionnement) et maintenace

	OM = 30     #???



	## Determination des puissance de chaque conmposants du systeme
	PV_P = p_npv    # est un scalaire deduit de l'optimisation PSO
	BAT_P = max_deficit_charge_reel  # puissance ou capacité maximale des batteries (scalaire)=== dimensionner poour une autom de deux jours
	DSL_P = Pdies_nom1  # puissance nominale d'un GE === 40kw

					# MAIN    Analyse Economique
	# -----------analyse economique du diesel.......

	# trouver les valeurs non nulles de la puissance generée par le GE pour evaleur le cout de fonctionnement du composant

	idxnonzero = np.nonzero(diesel)
	values_diesel_nonzero = diesel[idxnonzero]    # calcul des sommes des valeurs non nulles
	sum_diesel= np.sum(values_diesel_nonzero/Pdies_nom1 ) 		#calcul de la puissance totale fournie par le GE sur une annee
	# sum_diesel= math.floor(sum_diesel)
	fuel_cons = Fg * sum_diesel									# le fuel consommé
	duree_de_vie_B= DSL_LF // sum_diesel									#durée_de_cycle_vie_diesel 

	if duree_de_vie_B < PRJ_LF :
		nbre_remp = math.floor(PRJ_LF / duree_de_vie_B)
		price_diesel = DSL_C * DSL_P * nbre_remp
	else:
		price_diesel = DSL_C * DSL_LF

		#-----------------analyse Economique battery--------------------

	duree_de_vie_C =  math.floor(PRJ_LF / BAT_LF)
	price_battery = BAT_C * BAT_P * duree_de_vie_C

	# economic analysis
	i =  REAL_INTERST / 100         # Taux d'interet reel
	initial_cost = PV_C * PV_P + price_battery + price_diesel + PV_reg_C
	OM 			 = initial_cost * (OM/100)
	initial_cost = initial_cost + OM

	annual_cost  = initial_cost * ((i * (1+i)**PRJ_LF) / (((1+i)**PRJ_LF)-1))

	# i = REAL_INTERST /100
	annual_cost_fuel	= fuel_cons * PRJ_LF * ((i * (1+i)**PRJ_LF) / (((1+i)**PRJ_LF)-1)) 
	annual_cost 		=  annual_cost + annual_cost_fuel
	annual_load 		= np.sum(charge_jr)
	prix 				= annual_cost / annual_load

	# sum_values = values_nonzero.sum()

	# calcul du fuel consommé annuellement
	return prix, values_diesel_nonzero, sum_diesel

class Particule:
	"""creation d'une particule i avec ses differents attributs(
	poisiton, velocity, lpsp, dump, cout, rf)"""
	def __init__(self):
		
		self.position = np.zeros(4)
		self.velocity=np.zeros(4)
		self.lpsp= np.zeros(1)
		self.dump=np.zeros(1)
		self.cout=np.zeros(1)
		self.rf=np.zeros(1)
pi= Particule()   #appel a la classe Particule

class Swarm:
	"""Creaction du swarm (groupe) de n particules. chaque particule garde avec les memes attributs(
	poisiton, velocity, lpsp, dump, cout, rf) """
	def __init__(self,n):
		self.swarm_position= np.tile(pi.position, (n, 1))
		self.swarm_velocity = np.tile(pi.velocity, (n, 1))
		self.swarm_dump = np.tile(pi.dump, (n, 1))
		self.swarm_rf = np.tile(pi.rf, (n, 1))
		self.swarm_cout = np.tile(pi.cout, (n, 1))
		self.swarm_lpsp = np.tile(pi.lpsp, (n, 1))
sw=Swarm(5)			# appel a la classe Swarm  . l'argument 5 correspond à la population NPOP

class Best:
	"""Creation du best des attributs(
	poisiton, lpsp, dump, cout, rf) de la particule i"""
	def __init__(self):
		
		self.best_position = np.zeros(4)
		self.best_rf=np.zeros(1)
		self.best_lpsp=np.zeros(1)
		self.best_dump=np.zeros(1)
		self.best_cout=np.zeros(1)
bestpi=Best()

class Swarmbest:
	"""Creaction du bestswarm (groupe) de n particules. chaque best particule garde avec les memes attributs(
	poisiton, lpsp, dump, cout, rf) """

	def __init__(self,n):

		self.swarmbest_position= np.tile(bestpi.best_position, (n, 1))
		self.swarmbest_rf = np.tile(bestpi.best_rf, (n, 1))
		self.swarmbest_dump = np.tile(bestpi.best_dump, (n, 1))
		self.swarmbest_cout = np.tile(bestpi.best_cout, (n, 1))
		self.swarmbest_lpsp = np.tile(bestpi.best_lpsp, (n, 1))
swbest=Swarmbest(5)			# appel a la classe Swarm  . l'argument 5 correspond à la population NPOP

class Global:
	"""Creation du globalbest """
	def __init__(self):

		self.globalbest_position=np.zeros(4)


		self.globalbest_rf=np.inf
		self.globalbest_lpsp=np.inf
		self.globalbest_dump=np.inf
		self.globalbest_cout=np.inf
gb=Global()

#----------------Definition du problème
 #Initiatilisation des paramètres
C = 3                    #c 3 oût
W = 0.6                 # 0.15 Probabilité de pertes de charge
K = 0.99                #0.99 Facteur d'energie renouvelable
nvars = 1               #Nombre de systèmes


###Valeurs extrêmes
param=1 
Ppv_min=0  # minimum de puissance PV
Ppv_max=3500 #350 #200  # max de puissance PV
NPdies_min=1  #  nombre mini de générateurs diesel
NPdies_max=3  # 
AD_min=1  #  nombre Mini de jours d'autonomie du stockage
AD_max=3 
H_min=1  # nombre Mini de menage, groupe de consommateurs, nombre de villages, groupes de villages,...
H_max=15 


# definissons les limites des differentes grandeurs

LB = np.array([Ppv_min, AD_min, H_min, NPdies_min])
UB = np.array([Ppv_max, AD_max, H_max, NPdies_max])

max_iter=5
NPOP=5


ww=[]
	# ccc=4               #une valeur du coût
 #    ww=0.5              #une valeur de la Probabilité de pertes de charge
 #    kkk=2               #Facteur d'energie renouvelable
 #    ff=0                #initiatisation Compteur des itérations (nombre population)

    # cette loop permet d'effectuer les operations qu'elles engloubent pour chaque particule
for i in range(NPOP):
	ccc= 4
	ppp=0.5
	fff=2

	nnn=0

	sw.swarm_position[i][0]=np.random.randint(Ppv_min, Ppv_max)   #  size=(1,nvars)    bon a savoir les swarm que j'ai cree sont matrice_ligne alors 
	sw.swarm_position[i][1]=np.random.randint(AD_min,AD_max)		# normalement ils doivent etre des matrice_colonnes prend en compte les parametres nvars
	sw.swarm_position[i][2]=np.random.randint(H_min,H_max)
	sw.swarm_position[i][3]=np.random.randint(NPdies_min,NPdies_max)

	print()
	for j in range(4):
		sw.swarm_velocity[i][j]=np.random.rand()

	p_npv = sw.swarm_position[i][0]
	ad    = sw.swarm_position[i][1]
	# param = sw.swarm_position[i][2]    # nbre d'habitat (ou d'habitant)
	nPng  = round(sw.swarm_position[i][3])


# La matrice sw.swarm_pos de dim 5x4 obtenu peit erte interpreté comme suit: chaque ligne ou enregistrement represente un individu ou une particule et chaque colonne represente 
# respectivement p_npv , ad, param, et nPng. donc chaque indiv oub part est caracterisé par ces grandeurs 

###----------input ---read and handle files which contain databases ( irradiation , temperature etc.)

	contribution = np.zeros([6,100])
	Eb     = np.zeros([100])
	time1  = np.zeros([100])
	diesel = np.zeros([100])
	Edump  = np.zeros([100])
	Edch   = np.zeros([100])
	Ech    = np.zeros([100])
	Pdch   = np.zeros([100])
	Pch    = np.zeros([100])


			#---C-radiation--------#########
	radiation = "radiation_col.xlsx"
	df_radiation_sh0 = pd.read_excel(radiation, sheet_name=0,index_col=0)   # on appelle la feuille numero 0 (il ya qu'une seule feuille)

			#----temperature------######
	temperature = "temperature_baba.xlsx"
	df_temperature_sh0 = pd.read_excel(temperature, sheet_name=0)   # on peut ajouter l'argument index_col pour eliminer 
		
			#----charge===load------#####
	load = "charge.xlsx"
	df_load_sh0 = pd.read_excel(load, sheet_name=0)

			#----definition de la puissance de sortie du panneau photovoltaique
	t_amb = df_temperature_sh0.loc[:,'temp']         # selection d'une colonne en l'occurence temp
	g_solar=  df_radiation_sh0.radiation             # autre methode de selectionner une colonne
	# g_solar=g_solar.head(100)  


	t_amb1= 1 * t_amb
	g_ref = 1   			 			#KW/m2
	t_ref = 25  				 		#degres celsius
	kt= -0.0037       					#e-3                   ???
	tc = t_amb1+(0.0256)*(g_solar)   	# puisque la temperature influence le rendement des panneaux.  ???
	u_pv = 0.986
	# p_npv =1971                  		# kWc puissance totale des panneaux trouvée apres optimisation PSO (c'est un paramtre d'entree)
	# ad = 2							#nombre de jours d'autonomie (paramtre d'entree)
	# nPng= 2							#nbre de generateur (paramtre d'entree)	
		##-----sorties------
	P_pvout = u_pv * (p_npv * ( g_solar / g_ref ) * (1 + kt * ( tc - t_ref )))  # la puissance de sortie des panneaux 

	# P_pvout = P_pvout.head(100)    	# les 100 premieres heures 															  #p_npv la puissance crete des panneaux il deduit du pso. (voir main pso)
					
		#--------#Dimensionnement de la batterie---------------------------------------------

										# parametre des efficicaté de l'onduleur , batterie et l'etat de la batterie
	uinv=0.92
	ub  =0.85
	dod =0.8  				     		# depth of discharge 0.5 0.7 in the article 80%


	param=1 							# param represente le facteur multiplicatif de la charge , cest le nombre de menages ou de concession

	charge_Wh  = df_load_sh0.charge     # Wh  _____________________________>>> veri
	charge_kWh = charge_Wh * 0.001		#kWh
	# charge_jr  = charge_kWh.head(100) * param
	charge_jr  = charge_kWh * param
	sum_charge_jr = np.sum(charge_jr)
	#	describ_charge_jr= charge_jr.describe()
	max_deficit_charge_thoerique = max(abs(charge_jr - P_pvout))     #le max de la valeur de charge non satisfaite. il permet de dimmensionner les batteries. 
															   			#  !!! ici la charge_jr est tres elevé, elle est exprimee en Wh alors que la P_pvout en kW 
															   		#demande à M Tankari??????
	max_deficit_charge_reel = (max_deficit_charge_thoerique * ad ) // (uinv * ub * dod)	# capacité de charge max des batteries prenant en charge 2 jours d'autonomie

	Ebmax = max_deficit_charge_reel     #kWh	
	Ebmin = Ebmax * (1-dod)	
	SoCb = 1                            # state of charge
	Eb[0] = SoCb* Ebmax					#SoC at starting time ==== SoC à l'instant t=0
	    #-------definition des constante du Generator diesel-------#

	Pdies_nom1 = 40 #kW 
	Png = nPng * Pdies_nom1
	Png_min= 0 

	    # -----consommation de fuel 
	Bg = 0.08145
	Ag = 0.24006
	Fg= (Ag + Bg)* Png



	for t in range(1,100):
		if P_pvout[t] > (charge_jr[t]/ uinv) :

			if P_pvout[t] > (charge_jr[t]) :

				Ech,Eb ,Edump=charge(Puissance_pv= P_pvout, Energie_Batterie=Eb, Energie_Batterie_max= Ebmax, Puissance_charge= Pch, temps_iterative=t , Energie_flexible=Edump, Energie_charge=Ech,uinv=0.9)
				time1[t] =1
				contribution[0,t]= P_pvout[t]
				contribution[1,t]= Edch[t]
				contribution[2,t]= diesel[t]
				contribution[3,t]= Edump[t]
				contribution[4,t]= charge_kWh[t]
				contribution[5,t]= Eb[t]			
			else:
				Eb[t] = Eb[t-1]

			
		else:
			Edch, Eb, time1, Edump, diesel =decharge(Puissance_pv= P_pvout, Energie_Batterie=Eb, Energie_Batterie_max= Ebmax, Puissance_decharge= Pdch, temps_iterative=t , Puissance_generateur= Png , Energie_Batterie_min= Ebmin, Energie_flexible=Edump, diesel= diesel,time1= time1,uinv=0.9)

			contribution[0,t]= P_pvout[t]
			contribution[1,t]= Edch[t]
			contribution[2,t]= diesel[t]
			contribution[3,t]= Edump[t]
			contribution[4,t]= charge_jr[t]
			contribution[5,t]= Eb[t]



		df_contribution = pd.DataFrame(contribution)	
		data_contribution = df_contribution.transpose()   #ici je transpose pour avoir le temps , la P_pvout , Edch... en colonne
		data_contribution	= data_contribution.rename( columns={0: 'P_pvout', 1: "Edch", 2:'diesel', 3:'Edump', 4:'charge_jr', 5:'Eb'} )  # ici je renomme les colonnes 0, 1... par leur designation
		time= pd.date_range('1/1/2011', periods=100, freq='H')
		data_contribution['time']= time
		data_contribution= data_contribution.set_index('time')


		b = data_contribution.sum()     # ici je calcule la somme de chaque colonne

		b['renewable_factor'] = b['Edump']/(b['P_pvout']+b['Edch'])  # on calcule ensuite  facteur d'energie . on  ajoute une colonne qui lui sera destiné

		somme_charge_jr = b['charge_jr']

		
		total_loss = 0
		for t in range(100):
			if charge_jr[t] >  P_pvout[t] + Eb[t] - Ebmin + diesel[t]:
				total_loss = total_loss + (charge_jr[t] - (P_pvout[t] + Eb[t] - Ebmin + diesel[t]))

		LPSP = total_loss/somme_charge_jr
		prix ,values_diesel_nonzero, sum_diesel  = economic(diesel=diesel, charge_jr=charge_jr, Fg=Fg, Pdies_nom1=Pdies_nom1, max_deficit_charge_reel=max_deficit_charge_reel, p_npv=p_npv,  param=1) 
		

		# print('Pour particule {} à t = {} son prix est : {}'.format(i,t, prix))

		#+++++++++++++++================================================================================================================


	# affection des grandeurs LPSP , ... cout à chaque particule dans NPOP

	sw.swarm_lpsp[i]= LPSP
	sw.swarm_dump[i]= b['Edump']
	sw.swarm_rf[i]= b['renewable_factor']
	sw.swarm_cout[i]= prix



	# 
	swbest.swarmbest_position[i]= sw.swarm_position[i]

	swbest.swarmbest_lpsp[i]= sw.swarm_lpsp[i]
	swbest.swarmbest_rf[i]= sw.swarm_rf[i]
	swbest.swarmbest_cout[i]= sw.swarm_cout[i]
	swbest.swarmbest_dump[i]= sw.swarm_dump[i]

	#enregistrement des meilleurs attributs en fonction du cout minimal

	if swbest.swarmbest_cout[i] < gb.globalbest_cout:
		# gb=swbest[i]
		gb.globalbest_cout=swbest.swarmbest_cout[i]
		gb.globalbest_dump=swbest.swarmbest_dump[i]
		gb.globalbest_lpsp=swbest.swarmbest_lpsp[i]
		gb.globalbest_rf=swbest.swarmbest_rf[i]
		gb.globalbest_position=swbest.swarmbest_position[i]



####+++++=============================================================================================================================================================##
				##### 															Main du PSO    #################
#####+================================================================================================================================================================

Fmin= np.zeros([max_iter,1])
Xmin= np.zeros([max_iter,4])

for u in range(max_iter):
	vv= 0

	# print(u)
	for i in range(NPOP):
		prix = 4
		LPSP= 0.7
		rf= 2
# W=0.6


		bb=0
    
		while LPSP >= W:
		# prix >= C :
#		or LPSP >= W:
			w=0.5       # constant inertia weight (how much to weigh the previous velocity)
			c1=1        # cognative constant
			c2=2        # social constant

	        # for i in range(0,num_dimensions):
			r1=np.random.rand()
			r2=np.random.rand()

			for j in range(4):
				# print(j)
				sw.swarm_velocity[i][j]=w * sw.swarm_velocity[i][j] + (c1 * r1 * (swbest.swarmbest_position[i][j] - sw.swarm_position[i][j])) + c2 * r2 * (
				 																								gb.globalbest_position[j] - sw.swarm_position[i][j])
				sw.swarm_position[i][j] = sw.swarm_position[i][j] + sw.swarm_velocity[i][j]


				# sw.swarm_position[i][j]= max(sw.swarm_position[i][j] , LB(j))


			p_npv = round( sw.swarm_position[i][0])
			ad    = round( sw.swarm_position[i][1])
			# param = round( sw.swarm_position[i][2])    # nbre d'habitat (ou d'habitant)
			nPng  = round( sw.swarm_position[i][3])	

			# A partir de là on charge la fonction technoeconomic


###----------input ---read and handle files which contain databases ( irradiation , temperature etc.)

			contribution = np.zeros([6,100])
			Eb     = np.zeros([100])
			time1  = np.zeros([100])
			diesel = np.zeros([100])
			Edump  = np.zeros([100])
			Edch   = np.zeros([100])
			Ech    = np.zeros([100])
			Pdch   = np.zeros([100])
			Pch    = np.zeros([100])


					#----radiation--------#########
			radiation = "radiation_col.xlsx"
			df_radiation_sh0 = pd.read_excel(radiation, sheet_name=0,index_col=0)   # on appelle la feuille numero 0 (il ya qu'une seule feuille)

					#----temperature------######
			temperature = "temperature_baba.xlsx"
			df_temperature_sh0 = pd.read_excel(temperature, sheet_name=0)   # on peut ajouter l'argument index_col pour eliminer 
				
					#----charge===load------#####
			load = "charge.xlsx"
			df_load_sh0 = pd.read_excel(load, sheet_name=0)

					#----definition de la puissance de sortie du panneau photovoltaique
			t_amb = df_temperature_sh0.loc[:,'temp']         # selection d'une colonne en l'occurence temp
			g_solar=  df_radiation_sh0.radiation             # autre methode de selectionner une colonne
			# g_solar=g_solar.head(100)  


			t_amb1= 1 * t_amb
			g_ref = 1   			 			#KW/m2
			t_ref = 25  				 		#degres celsius
			kt= -0.0037       					#e-3                   ???
			tc = t_amb1+(0.0256)*(g_solar)   	# puisque la temperature influence le rendement des panneaux.  ???
			u_pv = 0.986


			# p_npv =1971                  		# kWc puissance totale des panneaux trouvée apres optimisation PSO (c'est un paramtre d'entree)
			# ad = 2							#nombre de jours d'autonomie (paramtre d'entree)
			# nPng= 2							#nbre de generateur (paramtre d'entree)	
				##-----sorties------
			P_pvout = u_pv * (p_npv * ( g_solar / g_ref ) * (1 + kt * ( tc - t_ref )))  # la puissance de sortie des panneaux 

			# P_pvout = P_pvout.head(100)    	# les 100 premieres heures 															  #p_npv la puissance crete des panneaux il deduit du pso. (voir main pso)
							
				#--------#Dimensionnement de la batterie---------------------------------------------
												# parametre des efficicaté de l'onduleur , batterie et l'etat de la batterie
			uinv=0.92
			ub  =0.85
			dod =0.8  				     		# depth of discharge 0.5 0.7 in the article 80%


			param=1 							# param represente le facteur multiplicatif de la charge , cest le nombre de menages ou de concession

			charge_Wh  = df_load_sh0.charge     # Wh  _____________________________>>> veri
			charge_kWh = charge_Wh * 0.001		#kWh
			# charge_jr  = charge_kWh.head(100) * param
			charge_jr  = charge_kWh * param
			sum_charge_jr = np.sum(charge_jr)
			#	describ_charge_jr= charge_jr.describe()
			max_deficit_charge_thoerique = max(abs(charge_jr - P_pvout))     #le max de la valeur de charge non satisfaite. il permet de dimmensionner les batteries. 
																	   			#  !!! ici la charge_jr est tres elevé, elle est exprimee en Wh alors que la P_pvout en kW 
																	   		#demande à M Tankari??????
			max_deficit_charge_reel = (max_deficit_charge_thoerique * ad ) // (uinv * ub * dod)	# capacité de charge max des batteries prenant en charge 2 jours d'autonomie

			Ebmax = max_deficit_charge_reel     #kWh	
			Ebmin = Ebmax * (1-dod)	
			SoCb = 1                            # state of charge
			Eb[0] = SoCb* Ebmax					#SoC at starting time ==== SoC à l'instant t=0
			    #-------definition des constante du Generator diesel-------#

			Pdies_nom1 = 40  #kW 
			Png = nPng * Pdies_nom1
			Png_min= 0 

			    # -----consommation de fuel 
			Bg = 0.08145
			Ag = 0.1006
			Fg= (Ag + Bg)* Png

#--------
			for t in range(1,100):
				if P_pvout[t] > (charge_jr[t]/ uinv) :

					if P_pvout[t] > (charge_jr[t]) :

						Ech,Eb ,Edump=charge(Puissance_pv= P_pvout, Energie_Batterie=Eb, Energie_Batterie_max= Ebmax, Puissance_charge= Pch, temps_iterative=t , Energie_flexible=Edump, Energie_charge=Ech,uinv=0.9)
						time1[t] =1
						contribution[0,t]= P_pvout[t]
						# print('ccontribution_pvout si la cond P_pvout[t] > (charge_jr[t]) (ok) \n à t={} on a {}'.format(t, contribution[0,t]))
						contribution[1,t]= Edch[t]
						contribution[2,t]= diesel[t]
						contribution[3,t]= Edump[t]
						contribution[4,t]= charge_kWh[t]
						contribution[5,t]= Eb[t]			
					else:
						Eb[t] = Eb[t-1]

					
				else:
					Edch, Eb, time1, Edump, diesel =decharge(Puissance_pv= P_pvout, Energie_Batterie=Eb, Energie_Batterie_max= Ebmax, Puissance_decharge= Pdch, temps_iterative=t , Puissance_generateur= Png , Energie_Batterie_min= Ebmin, Energie_flexible=Edump, diesel= diesel,time1= time1,uinv=0.9)

					contribution[0,t]= P_pvout[t]
					contribution[1,t]= Edch[t]
					contribution[2,t]= diesel[t]
					contribution[3,t]= Edump[t]
					contribution[4,t]= charge_jr[t]
					contribution[5,t]= Eb[t]

				df_contribution = pd.DataFrame(contribution)	
				data_contribution = df_contribution.transpose()   #ici je transpose pour avoir le temps , la P_pvout , Edch... en colonne
				data_contribution	= data_contribution.rename( columns={0: 'P_pvout', 1: "Edch", 2:'diesel', 3:'Edump', 4:'charge_jr', 5:'Eb'} )  # ici je renomme les colonnes 0, 1... par leur designation
				time= pd.date_range('1/1/2011', periods=100, freq='H')
				data_contribution['time']= time
				data_contribution= data_contribution.set_index('time')


				b = data_contribution.sum()     # ici je calcule la somme de chaque colonne

				b['renewable_factor'] = b['Edump']/(b['P_pvout']+b['Edch'])  # on calcule ensuite  facteur d'energie . on  ajoute une colonne qui lui sera destiné

				somme_charge_jr = b['charge_jr']

				# rf est important fait parti des return
				# rf= b['renewable_factor']    # à mettre ausssi dans les autres fonctions technoeconomic
				# Edump = b['Edump']


				total_loss = 0
				for t in range(100):
					if charge_jr[t] >  P_pvout[t] + Eb[t] - Ebmin + diesel[t]:
						total_loss = total_loss + (charge_jr[t] - (P_pvout[t] + Eb[t] - Ebmin + diesel[t]))

				# LPSP est important fait parti des retrurn
				LPSP = total_loss/somme_charge_jr

				# Prix auusi
				prix ,values_diesel_nonzero, sum_diesel  = economic(diesel=diesel, charge_jr=charge_jr, Fg=Fg, Pdies_nom1=Pdies_nom1, max_deficit_charge_reel=max_deficit_charge_reel, p_npv=p_npv,  param=1) 
				print('Le prix est : {} '.format(prix))
				print('Le LPSP est : {} '.format(LPSP))
				print()
				bb=bb+1


	# 	sw.swarm_lpsp[i]= LPSP
	# 	sw.swarm_dump[i]= b['Edump']
	# 	sw.swarm_rf[i]= b['renewable_factor']
	# 	sw.swarm_cout[i]= prix


	# 	if sw.swarm_cout[i] < swbest.swarmbest_cout[i]:
	# 		swbest.swarmbest_cout[i]= sw.swarm_cout[i]
	# 		swbest.swarmbest_rf[i]= sw.swarm_rf[i]
	# 		swbest.swarmbest_dump[i]= sw.swarm_dump[i]
	# 		swbest.swarmbest_position[i]= sw.swarm_position[i]

	# 		if swbest.swarmbest_cout[i] < gb.globalbest_cout:
	# 		# gb=swbest[i]
	# 			gb.globalbest_cout=swbest.swarmbest_cout[i]
	# 			gb.globalbest_dump=swbest.swarmbest_dump[i]
	# 			gb.globalbest_lpsp=swbest.swarmbest_lpsp[i]
	# 			gb.globalbest_rf=swbest.swarmbest_rf[i]
	# 			gb.globalbest_position=swbest.swarmbest_position[i]




	# Fmin[u]= gb.globalbest_dump
	# Xmin[u]= gb.globalbest_position
	# p_npv= round(gb.globalbest_position[0])
	# ad= round(gb.globalbest_position[1])
	# param= round(gb.globalbest_position[2])
	# nPng= round(gb.globalbest_position[3])

		




# print('rf \n{}'.format(rf))

					

print('gb.globalbest_position \n{}'.format(gb.globalbest_position))
print()
print('gb.globalbest_position \n{}'.format(gb.globalbest_position[0]))
print()

print('sw.swarm_velocity {}'.format(sw.swarm_velocity))
print()

	# print(swbest.swarmbest_cout)
print('gb.globalbest_cout :{}'.format(gb.globalbest_cout))






# print(sw)





# print('sw.swarm_position\n{}'.format(sw.swarm_position))
print()
print('swbest.swarmbest_position est \n {}'.format(swbest.swarmbest_position)) 
print()	
 
print('swbest.swarmbest_position[i][ 0]est \n {}'.format(swbest.swarmbest_position[i][0])) 
print()		

print('sw.swarn_lpsp est \n {}'.format(sw.swarm_lpsp) )  
# print('sw.swarn_dump est \n {}'.format(sw.swarm_dump) )
# print('sw.swarn_rf est \n {}'.format(sw.swarm_rf) )
print('sw.swarn_cout est \n {}'.format(sw.swarm_cout) )



print('data_contribution {}'.format(data_contribution.head()))
print()

print('type(data_contribution) {}'.format(type(data_contribution)))
print()

print('Prix  : {} Euros/kW'.format(prix))

print()

print('la somme des grandeurs b est  :\n{} '.format(b))  
print()

print("la somme des grandeurs b['Edump'] est  :\n{} ".format(b['Edump']))  
print()

print('sw.swarm_position:\n {}'.format(sw.swarm_position))
# print('la somme des grandeurs type(b) est  :\n{} '.format(type(b)))  
print()



print('p_npv:\n {}'.format(p_npv))

print()
print('ad:\n {}'.format(ad))

print()
print('param:\n {}'.format(param) )
print()


print('nPng:\n {}'.format(nPng)) 
print()


# print()





plt.figure()

data_contribution.P_pvout.plot()
data_contribution.charge_jr.plot()
data_contribution.diesel.plot()
data_contribution.Edump.plot()
# data_contribution.Edch.plot()
data_contribution.Eb.plot()
plt.legend()
		
plt.savefig('Flux de puissance3.png')
plt.show()




		