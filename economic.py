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
