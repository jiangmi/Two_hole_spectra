This program deals with two holes in the impuriey model. 

You have the Cu site (only one) and the O square lattice surrounding it. You start with both holes on the Cu, and then hopping will move them into states where there is one hole on the Cu and one on O, and also states where both holes are on the O. Again you should have a cutoff from how far away the holes can move from the Cu, and truncate contributions from states with holes farther out.

You have to index these allowed states and define how the Hamiltonian acts on them, and then diagonalize within this variational subspace with Lanczos.

Difference from the periodic lattice model calculation:
1. There is no k-dependence 
2. The impurity model rotates the O lattice by 45 degrees so that Cu locates in the middle of a square lattice. Then tpd is along diagonal direction and tpp is (1,0) or (0,1) etc.
 
