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
            resultats[i, j] = float(st.iloc[0])
    return resultats

res = (monte())
moy = np.mean(res, axis=0)
fig = plt.figure(figsize=(12,5))
ax1 = plt.subplot2grid((2,2), (0,0))
ax2 = plt.subplot2grid((2,2), (0,1))
ax3 = plt.subplot2grid((2,2), (1,0), colspan=2)
ax1.plot(moy, color='green')
highest_i = np.argmax(res[:,-1])
highest = res[highest_i,:] 
lowest_i = np.argmin(res[:,-1])
lowest = res[lowest_i,:]
ax1.plot(highest, color='blue')
ax1.plot(lowest,color='red')
#plt.fill_between(highest, lowest, color='gray', alpha=0.2)
ax1.set_title("Simulation moyenne, plus pessimiste et plus optimiste")
for courbe in res:
    ax3.plot(courbe)
ax3.set_title('Simulations de Monte Carlo')
final_val = res[:,-1]
taille = int(0.8*len(res))
seuil = np.partition(final_val,taille)[taille]
mask = final_val >= seuil
val_sel = res[mask]
print(len(val_sel))
for ele in val_sel:
    ax2.plot(ele)
ax2.set_title("20% plus optimistes simulations")

plt.tight_layout()
plt.show()