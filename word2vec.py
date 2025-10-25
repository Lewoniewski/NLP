import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

print("Loading pre-trained Word2vec model (this may take a minute)...")
print("Using Google News vectors (3 million words, 300 dimensions)")

# Load pre-trained Word2vec model
model = api.load('word2vec-google-news-300')

# Check which variations of team names exist in the model
print("\nChecking available terms in the model...")

terms_to_check = ["Germany","France","Berlin","Paris"]

available_terms = {}
for term in terms_to_check:
    if term in model:
        print(f"✓ '{term}' found")
        available_terms[term] = model[term]
    else:
        print(f"✗ '{term}' not found")

Germany = model['Germany']
France = model['France']
Berlin = model['Berlin']
Paris = model['Paris']

predicted = Paris - France + Germany


print("\nAnalogy: Paris - France + Germany = ?")
print("\nTop 25 most similar words to the result:")
similar = model.similar_by_vector(predicted, topn=25)
for word, similarity in similar:
    print(f"  {word}: {similarity:.4f}")



# Create visualization with actual Word2vec vectors
print("\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

# Select terms for visualization (use what's available)
viz_terms = []
viz_labels = []

# Try related terms
termz=["Germany","France","Berlin","Paris","Munich","Cologne","Düsseldorf","Stuttgart","Marseille","Lyon","Toulouse","Hamburg","Strasbourg","Montpellier","Bordeaux","Frankfurt","Leipzig","Dortmund","Essen","Bremen","Hanover","Nice","Poland","Warsaw","Poznań","Gdańsk","Wrocław","Kraków","Katowice","Gdańsk","Łódź","Lublin","Białystok"]


main_terms = []
for tt in termz:
    main_terms.append((tt,tt))

for term, label in main_terms:
    if term in model:
        viz_terms.append(model[term])
        viz_labels.append(label)


# Also add the classic king-queen example
classic_terms = [
    ('king', 'king'),
    ('queen', 'queen'),
    ('man', 'man'),
    ('woman', 'woman')
]

for term, label in classic_terms:
    if term in model:
        viz_terms.append(model[term])
        viz_labels.append(label)
        termz.append(term)

# Convert to numpy array and reduce dimensions with PCA
vectors_matrix = np.array(viz_terms)
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors_matrix)

# Create the visualization
fig, ax1 = plt.subplots(figsize=(16, 7))

# Plot 1: Sports teams (if available)
sports_count = len([l for l in viz_labels if l in termz])

for i, label in enumerate(viz_labels[:sports_count]):
    x, y = vectors_2d[i]
    if label in ['Germany', 'France', "Poland"]:
        color, marker, siz, alf, fnsiz, xtex, ytex, fontw = 'blue', 's', 300, 0.9, 13, 10, -3, 'bold'
    elif label in ['Berlin', 'Paris', "Warsaw"]:
        color, marker, siz, alf, fnsiz, xtex, ytex, fontw = 'red', 'o', 200, 0.6, 13, 10, -3, 'bold'
    elif label in ['king', 'queen', "man", "woman"]:
        color, marker, siz, alf, fnsiz, xtex, ytex, fontw = 'grey', '^', 200, 0.6, 13, 10, -3, 'normal'
    else:
        color, marker, siz, alf, fnsiz, xtex, ytex, fontw = 'orange', 'o', 200, 0.6, 13, 10, -3, 'normal'
    
    ax1.scatter(x, y, c=color, marker=marker, s=siz, alpha=alf)
    ax1.annotate(label, (x, y), xytext=(xtex, ytex), textcoords='offset points',
                fontsize=fnsiz, fontweight=fontw)
classic_indices = [i for i, l in enumerate(viz_labels) if l in termz]
classic_vectors_2d = vectors_2d[classic_indices]
classic_labels_list = [viz_labels[i] for i in classic_indices]
idx_map = {label: idx for idx, label in enumerate(classic_labels_list)}

for xx in [('France','Paris'),('Germany','Berlin'),('Poland','Warsaw')]:
    ax1.annotate('', xy=classic_vectors_2d[idx_map[xx[0]]], 
                xytext=classic_vectors_2d[idx_map[xx[1]]],
                arrowprops=dict(arrowstyle='<-', color='green', lw=2.5, alpha=0.6))

ax1.set_title('Word2vec Vectors: Cities & Countries (Reduced to 2D via PCA)', 
                fontsize=12, fontweight='bold')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=10)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=10)
ax1.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Note: Original vectors are 300-dimensional.")
print("We used PCA to reduce them to 2D for visualization.")
print("The actual Word2vec model operates in the full 300D space!")
print("="*60)



# --- PCA to 3D ---
vectors_matrix = np.array(viz_terms)
pca = PCA(n_components=3, random_state=0)
vectors_3d = pca.fit_transform(vectors_matrix)

# Helper indices (reusing your variables)
sports_count = len([l for l in viz_labels if l in termz])
classic_indices = [i for i, l in enumerate(viz_labels) if l in termz]
classic_vectors_3d = vectors_3d[classic_indices]
classic_labels_list = [viz_labels[i] for i in classic_indices]
idx_map = {label: idx for idx, label in enumerate(classic_labels_list)}

# --- Create 3D figure ---
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([2, 1, 1])
# Scatter: same logic as before (teams/cities first if you like)
for i, label in enumerate(viz_labels[:sports_count]):
    x, y, z = vectors_3d[i]
    # Keep your color/marker scheme:
    if label in ['Germany', 'France', 'Poland']:
        color, marker, rozm, fontw = 'blue', 's', 250, 'bold'
    elif label in ['Berlin', 'Paris', 'Warsaw']:
        color, marker, rozm, fontw = 'red', 'o', 150, 'bold'
    elif label in ['king', 'queen', "man", "woman"]:
        color, marker, rozm, fontw = 'grey', '^', 150, 'normal'
    else:
        color, marker, rozm, fontw = 'orange', 'o', 150, 'normal'
    ax.scatter(x, y, z, c=color, marker=marker, s=rozm, alpha=0.8)
    ax.text(x, y, z, f' {label}', fontsize=12,fontweight=fontw)

# Arrows for city→capital (e.g., Paris <- France, Berlin <- Germany)
def arrow_3d(ax, start, end, lw=2.0, alpha=0.7):
    # draw a vector from end to start (consistent with your '<-' style)
    sx, sy, sz = end
    dx, dy, dz = (start - end)
    ax.quiver(sx, sy, sz, dx, dy, dz, arrow_length_ratio=0.1, linewidth=lw, alpha=alpha, color='green')

for pair in [('France','Paris'), ('Germany','Berlin'), ('Poland','Warsaw')]:
    a, b = pair
    if a in idx_map and b in idx_map:
        A = classic_vectors_3d[idx_map[a]]
        B = classic_vectors_3d[idx_map[b]]
        arrow_3d(ax, start=A, end=B)

# Labels, title, grid

ax.set_title('Word2Vec Vectors: Cities & Countries (PCA to 3D)', fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=10)
ax.grid(True)

# Make axes roughly equal for better geometry perception (Matplotlib ≥ 3.3)
try:
    ax.set_box_aspect([2, 1, 1])
except Exception:
    pass  # older Matplotlib will ignore

plt.tight_layout()
plt.show()
