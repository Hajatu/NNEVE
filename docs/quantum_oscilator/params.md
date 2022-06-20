# Quantum oscilator parameters

## Introduction

QOParams class is responsible for holding variables which can change during the
learning of QONetwork.

Currenly there is only one attribute of QOParams object:

- **c** _(float)_ _- while searching for eigenvalues, learning algorithm
  increases value of c by 0.16 every genration, which increases Ldrive. That
  forces the network to search for larger eigenvalues and the associated
  eigenfunctions_

Value update is done in **.update()** method.
