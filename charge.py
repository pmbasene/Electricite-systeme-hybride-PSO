def charge(Puissance_pv, Energie_Batterie, Energie_Batterie_max , Puissance_charge, temps_iterative , Energie_flexible, Energie_charge,uinv=0.9): 

	Pch[t]= P_pvout [t]- (charge_jr[t]//uinv)     # la puissance residuelle 
												  #pour eventuellement charger les batteries
	Ech[t] = Pch[t]

	if Ech[t] <= Ebmax - Eb[t]:       #on fixe quand est ce qu'on doit autoriser la charge des batteries

		Eb[t] = Eb[t-1]  + Ech[t]		# la charge de la batterie à l'instant t+1 s
										#i la condition elle est remplie									       
		if Eb[t] > Ebmax: 

			Edump[t] = Ech[t] - (Ebmax - Eb[t])

			Eb[t] = Ebmax 
		else: 
			Edump[t] = 1            
	else: 
		Eb[t] = Ebmax 

		Edump[t]= Ech[t] - (Ebmax - Eb[t])

	return Ech,	Eb ,Edump