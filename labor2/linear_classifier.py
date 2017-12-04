import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datagenerator import generate_data

np.random.seed(2809)

#
#
#
#  binden Sie ihre myGrid Funktion ein um plotten zukönnen
#
#
#
   
def score(x_i, W):
    return W.dot(x_i)

def L(X, y, W):
    loss = 0    
    for x_i, y_i in zip(X[:,], y[:]):
        l_i = L_i_softmax(x_i, y_i, W)  # tauschen Sie diesen Aufruf...
        #
        #
        #
        #  l_i = L_i_hinge(x_i, y_i, W) # ...gegen diesen
        #
        #
        #
        loss += l_i
    loss = loss / X.shape[0]
    
    # regularizer
    loss += 0.1 * np.linalg.norm(W, ord=2)
    
    return loss
    
    
def L_i_softmax(x_i, y_i, W):
    s_i = score(x_i, W)
    e_i = np.exp(s_i)
    e_i_norm = e_i / np.sum(e_i)
    loss_i = -1.0 * np.log(e_i_norm[y_i])
    
    return loss_i
    

    
def eval_gradient(X, y, W):
    Lvalue = L(X, y, W)                       # L berechen
    gradient = np.zeros_like(W)
    delta = 0.00001                     # Delta festlegen
    
    w_iterator = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not w_iterator.finished:
        index =  w_iterator.multi_index
        W_at_index = W[index]           # Den aktuellen Wert von W speichern
        W[index] = W_at_index + delta   # Delta anwenden
        newL = L(X, y, W)                     # Funktionswert von L(W+delta) bestimmen
        gradient[index] = (newL - Lvalue)/delta  # Komponente berechen
        
        W[index] = W_at_index           # Ursprünglichen Wert in W wiederherstellen 
        w_iterator.iternext()           # In der nächsten Dimension weitermachen
        
    return gradient
    
def update_W(gradient, W):
    W = W - gradient * lr
    return W 
    
def predict(X, W):
    y_pred = []    
    for x_i in X[:,]:
         y_pred.append(np.argmax(score(x_i, W)))
         
    y_pred = np.array(y_pred)
    return y_pred
    


    
def train(X_train, y_train, X_val, y_val, W, iterations=100):

        
    for i in range(iterations):

        grad = eval_gradient(X_train, y_train, W)
        W = update_W(grad, W)
        
        y_pred = predict(X_val, W)
        
        accuracy = np.mean(np.array(y_pred == y_val, dtype=np.uint))
        print( 'Accuracy: ', accuracy)
        
        #
        #
        #plot_predict_grid(X_train, y_train, X_val, y_val, W, i)  #Hier einkommentieren!
        #
        #
    return W

    
def plot_predict_grid(X_train, y_train, X_val, y_val, W, i):
    xmin, xmax = X_train.min(), X_train.max()
    #
    # Aufruf ihrer Funktion aus der vorherigen Laborübung!
    #
    meshgrid = myGrid(xmin, xmax, 100)                       
    #
    #
    #
    meshgrid = np.hstack((meshgrid, np.ones((meshgrid.shape[0], 1))))    
    plt.figure(1)        
    plt.cla()
    grid_pred = predict(meshgrid, W)
    colors = ['red' if p==0 else 'blue' for p in grid_pred]
    plt.scatter(meshgrid[:,0], meshgrid[:,1], c=colors, alpha=0.1, edgecolors=None)
    colors = ['red' if y_i==0 else 'blue' for y_i in y_val]        
    plt.scatter(X_val[:,0], X_val[:,1], c=colors)
    plt.savefig('results/%d_result.png' % i)       
        

    
    
# Set global learning rate
lr = 0.1

def main():
    # Load data
    X, y = generate_data()
    
    # Get features and classes
    n_features = X.shape[1]
    n_classes = len(np.unique(y))    
    
    # Add a ones row to X (bias calculation)
    X = np.hstack((X, np.ones((X.shape[0], 1))))   
    
    # Split into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    
    # Initialize weigths and bias        
    W = np.hstack((np.random.randn(n_classes, n_features), 
                            np.zeros((n_classes, 1))))
    
        
    # Start training for N iterations
    W = train(X_train, y_train, X_val, y_val, W, iterations=50)
    
    

if __name__=="__main__":
    main()