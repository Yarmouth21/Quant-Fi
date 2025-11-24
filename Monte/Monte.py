import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def monte(tracker='AAPL', simulations=200, days=252):
    # Téléchargement des données historiques
    data = yf.download(tracker, period='1y', auto_adjust=False)
    returns = data['Adj Close'].pct_change().dropna()
    std_dev = returns.std()
    spot = data['Adj Close'].iloc[-1]
    variance = std_dev ** 2
    resultats = np.zeros((simulations, days))
    for i in range(simulations):
        st = spot
        for j in range(days):
            st = st * np.exp((returns.mean() - 0.5 * variance) + std_dev *np.random.normal(0,1))
            resultats[i, j] = st
    return resultats

res = (monte())
moy = np.mean(res, axis=0)
plt.figure(figsize=(12,5))
plt.subplot(2,1,2)
plt.plot(moy, color='red')
highest_i = np.argmax(res[:,-1])
highest = res[highest_i,:] 
lowest_i = np.argmin(res[:,-1])
lowest = res[lowest_i,:]
plt.plot(highest, color='blue')
plt.plot(lowest,color='red')
#plt.fill_between(highest, lowest, color='gray', alpha=0.2)
plt.title("Simulation moyenne et la plus optimiste")
plt.subplot(4,2,1)
for courbe in res:
    plt.plot(courbe)
plt.title('Simulations de Monte Carlo')
#plt.subplot(2,1,2)
seuil_haut = int(0.2*len(res))
#val = np.sort(res)[seuil_haut]
#val_sel = res[res >= seuil_haut]
#for ele in val_sel:
#    plt.plot(ele)
plt.show()