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