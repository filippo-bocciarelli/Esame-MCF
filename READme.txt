

---- MACCHINA DI GALTON -----

Per far partire la simulazione bisogna eseguire il codice e scegliere se si vuole effettuare una simulazione in 2D o in 3D. Per il 2D i parametri da inserire sono:

-p, la probabilità che la pallina faccia un passo verso destra
-N_p, il numero della palline
-n, il numero di passi che compie la pallina.

Il programma mostra l'istogramma della distribuzione della palline calcolando media e deviazione standard, esegue un fit dell'istogramma con la distribuzione binomiale e con quella gaussiana, restituendo per ciascuna il chi quadro.
E' possibile effettuare più simulazioni contemporaneamente inserendo più valori per uno dei parametri, se i parametri che variano sono p o n alla fine della simulazione viene mostrato un plot contenente le funzioni binomiali usate per il fit dei vari istogrammi.
Per inserire più valori per il numero di palline è necessario premere 1 quando richiesto, per più valori del numero di scelte va premuto 2 e per inserire più valori per la probabilità va premuto 3.

Il range di ciascun istogramma va da 0 a n+1, ciascun range corrisponde al numero di 'buche' che dovrebbe essere presenti in una macchina di Galton reale. 

---- Simulazione 3D -----


I parametri da inserire sono:
-px, la probabilità che la pallina faccia un passo verso destra nell'asse X
-py, la probabilità che la pallina faccia un passo verso destra nell'asse Y
-N_p, il numero della palline
-n, il numero di passi che compie la pallina

Anche qui si possono effettuare più simulazioni contemporaneamente inserendo molteplici valori per una delle varabili, il procedimento è analogo a quello sopra descritto con gli stessi numeri per modificare il numero di palline (1), le scelte (2) o la probabilità (3). 
Si devono inserire 2 parametri se si sceglie di modificare la probabilità in quanto in questo caso bisogna inserire sia i valori per px che quelli per py.
Se vengono inseriti più valori o per la probabilità o per n, alla fine dei plot degli istogrammi e dei fit con una gaussiana e una binomiale vengono restituiti 2 grafici, contenenti le funzioni binomiali di fit utilizzate corrispondenti ai rispettivi parametri. 

Il programma restituisce inoltre un'istogramma 2D che permette di vedere graficamente la posizione della palline sulla macchina di Galton 3D. Questo approccio è stato preferito ad un'istogramma 3D.

Volendo effettuare la simulazione di una macchina di Galton reale non ho ritenuto necessario scrivere un codice che supportasse n maggiore di 1000, in quanto questo è un numero più che sufficiente simulare un comportamento reale. Numeri maggiori possono essere utilizzati soltanto modificando il codice e implementando moduli (ad esempio Decimal) che supportano rappresentazioni di numeri molto grandi (9E+1000000000000000000), cosi' facendo però il programma impiega tempi molto più lunghi per la simulazione di dati anche con un numero basso di n.
Si può modificare il codice per utilizzare alti valori di n senza un maggior carico computazionale solo rinunciando al fit con la distribuzione binomiale, cosa che ho preferito non fare.


