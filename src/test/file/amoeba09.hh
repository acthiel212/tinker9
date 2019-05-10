namespace tinker_6fe8e913fe4da3d46849d10248ad2a4872b4da93 {
const char* amoeba09_prm =
R"**(
      ##############################
      ##                          ##
      ##  Force Field Definition  ##
      ##                          ##
      ##############################


forcefield              AMOEBA-2009

bond-cubic              -2.55
bond-quartic            3.793125
angle-trigonal          IN-PLANE
angle-cubic             -0.014
angle-quartic           0.000056
angle-pentic            -0.0000007
angle-sextic            0.000000022
opbendtype              ALLINGER
opbend-cubic            -0.014
opbend-quartic          0.000056
opbend-pentic           -0.0000007
opbend-sextic           0.000000022
torsionunit             0.5
vdwtype                 BUFFERED-14-7
radiusrule              CUBIC-MEAN
radiustype              R-MIN
radiussize              DIAMETER
epsilonrule             HHG
dielectric              1.0
polarization            MUTUAL
vdw-12-scale            0.0
vdw-13-scale            0.0
vdw-14-scale            1.0
vdw-15-scale            1.0
mpole-12-scale          0.0
mpole-13-scale          0.0
mpole-14-scale          0.4
mpole-15-scale          0.8
polar-12-scale          0.0
polar-13-scale          0.0
polar-14-scale          1.0
polar-15-scale          1.0
polar-12-intra          0.0
polar-13-intra          0.0
polar-14-intra          0.5
polar-15-intra          1.0
direct-11-scale         0.0
direct-12-scale         1.0
direct-13-scale         1.0
direct-14-scale         1.0
mutual-11-scale         1.0
mutual-12-scale         1.0
mutual-13-scale         1.0
mutual-14-scale         1.0


      #############################
      ##                         ##
      ##  Literature References  ##
      ##                         ##
      #############################


P. Ren and J. W. Ponder, "Polarizable Atomic Multipole Intermolecular
Potentials for Small Organic Molecules", in preparation.

J. W. Ponder and D. A. Case, "Force Fields for Protein Simulation",
Adv. Prot. Chem., 66, 27-85 (2003)

P. Ren and J. W. Ponder, "Polarizable Atomic Multipole Water Model for
Molecular Mechanics Simulation", J. Phys. Chem. B, 107, 5933-5947 (2003)

P. Ren and J. W. Ponder, "A Consistent Treatment of Inter- and
Intramolecular Polarization in Molecular Mechanics Calculations",
J. Comput. Chem., 23, 1497-1506 (2002)

Monovalent ion parameters taken from Zhi Wang, Ph.D. thesis, Department
of Chemistry, Washington University in St. Louis, May 2018; available
from https://dasher.wustl.edu/ponder/


      #############################
      ##                         ##
      ##  Atom Type Definitions  ##
      ##                         ##
      #############################


atom          1    1    He    "Helium Atom He"               2     4.003    0
atom          2    2    Ne    "Neon Atom Ne"                10    20.179    0
atom          3    3    Ar    "Argon Atom Ar"               18    39.948    0
atom          4    4    Kr    "Krypton Atom Kr"             36    83.800    0
atom          5    5    Xe    "Xenon Atom Xe"               54   131.290    0
atom          6    6    Li+   "Lithium Ion Li+"              3     6.941    0
atom          7    7    Na+   "Sodium Ion Na+"              11    22.990    0
atom          8    8    K+    "Potassium Ion K+"            19    39.098    0
atom          9    9    Rb+   "Rubidium Ion Rb+"            37    85.468    0
atom         10   10    Cs+   "Cesium Ion Cs+"              55   132.905    0
atom         11   11    Mg+   "Magnesium Ion Mg+2"          12    24.305    0
atom         12   12    Ca+   "Calcium Ion Ca+2"            20    40.078    0
atom         13   13    Zn+   "Zinc Ion Zn+2"               30    65.390    0
atom         14   14    F-    "Fluoride Ion F-"              9    18.998    0
atom         15   15    Cl-   "Chloride Ion Cl-"            17    35.453    0
atom         16   16    Br-   "Bromide Ion Br-"             35    79.904    0
atom         17   17    I-    "Iodide Ion I-"               53   126.904    0
atom         18   18    C     "Cyanide Ion C"                6    12.011    1
atom         19   19    N     "Cyanide Ion N"                7    14.007    1
atom         20   20    B     "Tetrafluoroborate B"          5    10.810    4
atom         21   21    F     "Tetrafluoroborate F"          9    18.998    1
atom         22   22    P     "Hexafluorophosphate P"       15    30.974    6
atom         23   23    F     "Hexafluorophosphate F"        9    18.998    1
atom         24   24    N     "Dinitrogen N2"                7    14.007    1
atom         25   25    C     "Methane CH4"                  6    12.011    4
atom         26   26    H     "Methane H4C"                  1     1.008    1
atom         27   27    C     "Ethane CH3"                   6    12.011    4
atom         28   28    H     "Ethane H3C"                   1     1.008    1
atom         29   27    C     "Alkane CH3-"                  6    12.011    4
atom         30   28    H     "Alkane H3C-"                  1     1.008    1
atom         31   29    C     "Alkane -CH2-"                 6    12.011    4
atom         32   30    H     "Alkane -H2C-"                 1     1.008    1
atom         33   31    C     "Alkane >CH-"                  6    12.011    4
atom         34   32    H     "Alkane -HC<"                  1     1.008    1
atom         35   33    C     "Alkane >C<"                   6    12.011    4
atom         36   34    O     "Water O"                      8    15.999    2
atom         37   35    H     "Water H"                      1     1.008    1
atom         38   36    O     "Methanol O"                   8    15.999    2
atom         39   37    H     "Methanol HO"                  1     1.008    1
atom         40   38    C     "Methanol CH3"                 6    12.011    4
atom         41   39    H     "Methanol H3C"                 1     1.008    1
atom         42   36    O     "Ethanol O"                    8    15.999    2
atom         43   37    H     "Ethanol HO"                   1     1.008    1
atom         44   40    C     "Ethanol CH2"                  6    12.011    4
atom         45   39    H     "Ethanol H2C"                  1     1.008    1
atom         46   40    C     "Ethanol CH3"                  6    12.011    4
atom         47   41    H     "Ethanol H3C"                  1     1.008    1
atom         48   40    C     "Propanol Me-CH2"              6    12.011    4
atom         49   42    H     "Propanol Me-CH2"              1     1.008    1
atom         50   40    C     "Propanol CH3"                 6    12.011    4
atom         51   41    H     "Propanol H3C"                 1     1.008    1
atom         52   36    O     "isoPropanol O"                8    15.999    2
atom         53   37    H     "isoPropanol HO"               1     1.008    1
atom         54   43    C     "isoPropanol >CH-"             6    12.011    4
atom         55   39    H     "isoPropanol >CH-"             1     1.008    1
atom         56   40    C     "isoPropanol CH3"              6    12.011    4
atom         57   41    H     "isoPropanol H3C"              1     1.008    1
atom         58   36    O     "Methyl Ether O"               8    15.999    2
atom         59   40    C     "Methyl Ether CH3"             6    12.011    4
atom         60   44    H     "Methyl Ether H3C"             1     1.008    1
atom         61   45    N     "Ammonia N"                    7    14.007    3
atom         62   46    H     "Ammonia H3N"                  1     1.008    1
atom         63   47    N     "Ammonium Ion N+"              7    14.007    4
atom         64   48    H     "Ammonium Ion H4N+"            1     1.008    1
atom         65   49    N     "Methyl Amine N"               7    14.007    3
atom         66   50    H     "Methyl Amine H2N"             1     1.008    1
atom         67   40    C     "Methyl Amine CH3"             6    12.011    4
atom         68   51    H     "Methyl Amine H3C"             1     1.008    1
atom         69   49    N     "Ethyl Amine N"                7    14.007    3
atom         70   50    H     "Ethyl Amine H2N"              1     1.008    1
atom         71   40    C     "Ethyl Amine CH2"              6    12.011    4
atom         72   51    H     "Ethyl Amine H2C"              1     1.008    1
atom         73   40    C     "Ethyl Amine CH3"              6    12.011    4
atom         74   41    H     "Ethyl Amine H3C"              1     1.008    1
atom         75   40    C     "Propyl Amine CH3"             6    12.011    4
atom         76   41    H     "Propyl Amine H3C"             1     1.008    1
atom         77   40    C     "Propyl Amine Me-CH2"          6    12.011    4
atom         78   42    H     "Propyl Amine Me-CH2"          1     1.008    1
atom         79   49    N     "Dimethyl Amine N"             7    14.007    3
atom         80   50    H     "Dimethyl Amine HN"            1     1.008    1
atom         81   40    C     "Dimethyl Amine CH3"           6    12.011    4
atom         82   51    H     "Dimethyl Amine H3C"           1     1.008    1
atom         83   49    N     "Trimethyl Amine N"            7    14.007    3
atom         84   40    C     "Trimethyl Amine CH3"          6    12.011    4
atom         85   51    H     "Trimethyl Amine H3C"          1     1.008    1
atom         86   49    N     "Pyrrolidine N"                7    14.007    3
atom         87   50    H     "Pyrrolidine HN"               1     1.008    1
atom         88   40    C     "Pyrrolidine C-CH2-C"          6    12.011    4
atom         89   52    H     "Pyrrolidine C-CH2-C"          1     1.008    1
atom         90   40    C     "Pyrrolidine CH2-N"            6    12.011    4
atom         91   52    H     "Pyrrolidine H2C-N"            1     1.008    1
atom         92   49    N     "NMePyrrolidine N"             7    14.007    3
atom         93   40    C     "NMePyrrolidine CH2-N"         6    12.011    4
atom         94   52    H     "NMePyrrolidine H2C-N"         1     1.008    1
atom         95   40    C     "NMePyrrolidine CH2<"          6    12.011    4
atom         96   52    H     "NMePyrrolidine H2C<"          1     1.008    1
atom         97   40    C     "NMePyrrolidine CH3-N"         6    12.011    4
atom         98   42    H     "NMePyrrolidine H3C-N"         1     1.008    1
atom         99   53    C     "Formamide C=O"                6    12.011    3
atom        100   54    H     "Formamide HCO"                1     1.008    1
atom        101   55    O     "Formamide O"                  8    15.999    1
atom        102   56    N     "Formamide N"                  7    14.007    3
atom        103   57    H     "Formamide H2N"                1     1.008    1
atom        104   53    C     "Acetamide C=O"                6    12.011    3
atom        105   55    O     "Acetamide O"                  8    15.999    1
atom        106   56    N     "Acetamide N"                  7    14.007    3
atom        107   57    H     "Acetamide H2N"                1     1.008    1
atom        108   40    C     "Acetamide CH3"                6    12.011    4
atom        109   58    H     "Acetamide H3C"                1     1.008    1
atom        110   40    C     "Propamide Me-CH2"             6    12.011    4
atom        111   42    H     "Propamide Me-CH2"             1     1.008    1
atom        112   40    C     "Propamide CH3"                6    12.011    4
atom        113   41    H     "Propamide H3C"                1     1.008    1
atom        114   53    C     "NMeFormamide C=O"             6    12.011    3
atom        115   54    H     "NMeFormamide HCO"             1     1.008    1
atom        116   55    O     "NMeFormamide O"               8    15.999    1
atom        117   56    N     "NMeFormamide N"               7    14.007    3
atom        118   57    H     "NMeFormamide HN"              1     1.008    1
atom        119   40    C     "NMeFormamide CH3"             6    12.011    4
atom        120   59    H     "NMeFormamide H3C"             1     1.008    1
atom        121   40    C     "NEtFormamide CH2-N"           6    12.011    4
atom        122   40    C     "NEtFormamide CH3-"            6    12.011    4
atom        123   41    H     "NEtFormamide H3C"             1     1.008    1
atom        124   53    C     "NMeAcetamide C=O"             6    12.011    3
atom        125   55    O     "NMeAcetamide O"               8    15.999    1
atom        126   56    N     "NMeAcetamide N"               7    14.007    3
atom        127   57    H     "NMeAcetamide HN"              1     1.008    1
atom        128   40    C     "NMeAcetamide CH3-N"           6    12.011    4
atom        129   59    H     "NMeAcetamide H3C-N"           1     1.008    1
atom        130   40    C     "NMeAcetamide CH3-C"           6    12.011    4
atom        131   58    H     "NMeAcetamide H3C-C"           1     1.008    1
atom        132   53    C     "DiMeFormamide C=O"            6    12.011    3
atom        133   54    H     "DiMeFormamide HCO"            1     1.008    1
atom        134   55    O     "DiMeFormamide O"              8    15.999    1
atom        135   56    N     "DiMeFormamide N"              7    14.007    3
atom        136   40    C     "DiMeFormamide CH3"            6    12.011    4
atom        137   59    H     "DiMeFormamide H3C"            1     1.008    1
atom        138   53    C     "DiMeAcetamide C=O"            6    12.011    3
atom        139   55    O     "DiMeAcetamide O"              8    15.999    1
atom        140   56    N     "DiMeAcetamide N"              7    14.007    3
atom        141   40    C     "DiMeAcetamide CH3-N"          6    12.011    4
atom        142   59    H     "DiMeAcetamide H3C-N"          1     1.008    1
atom        143   40    C     "DiMeAcetamide CH3-C"          6    12.011    4
atom        144   58    H     "DiMeAcetamide H3C-C"          1     1.008    1
atom        145   60    O     "Formic Acid OH"               8    15.999    2
atom        146   61    H     "Formic Acid HO"               1     1.008    1
atom        147   62    C     "Formic Acid C=O"              6    12.011    3
atom        148   55    O     "Formic Acid O=C"              8    15.999    1
atom        149   63    H     "Formic Acid HC=O"             1     1.008    1
atom        150   60    O     "Acetic Acid OH"               8    15.999    2
atom        151   61    H     "Acetic Acid HO"               1     1.008    1
atom        152   62    C     "Acetic Acid C=O"              6    12.011    3
atom        153   55    O     "Acetic Acid O=C"              8    15.999    1
atom        154   40    C     "Acetic Acid CH3"              6    12.011    4
atom        155   64    H     "Acetic Acid H3C"              1     1.008    1
atom        156   62    C     "Formaldehyde C=O"             6    12.011    3
atom        157   55    O     "Formaldehyde O=C"             8    15.999    1
atom        158   63    H     "Formaldehyde HC=O"            1     1.008    1
atom        159   62    C     "Acetaldehyde C=O"             6    12.011    3
atom        160   55    O     "Acetaldehyde O=C"             8    15.999    1
atom        161   40    C     "Acetaldehyde CH3"             6    12.011    4
atom        162   64    H     "Acetaldehyde H3C"             1     1.008    1
atom        163   63    H     "Acetaldehyde HC=O"            1     1.008    1
atom        164   65    S     "Hydrogen Sulfide S"          16    32.066    2
atom        165   66    H     "Hydrogen Sulfide H2S"         1     1.008    1
atom        166   67    S     "Methyl Sulfide S"            16    32.066    2
atom        167   68    H     "Methyl Sulfide HS"            1     1.008    1
atom        168   62    C     "Methyl Sulfide CH3"           6    12.011    4
atom        169   69    H     "Methyl Sulfide H3C"           1     1.008    1
atom        170   67    S     "Dimethyl Sulfide S"          16    32.066    2
atom        171   62    C     "Dimethyl Sulfide CH3"         6    12.011    4
atom        172   69    H     "Dimethyl Sulfide H3C"         1     1.008    1
atom        173   67    S     "Dimethyl Disulfide S"        16    32.066    2
atom        174   62    C     "Dimethyl Disulfide CH3"       6    12.011    4
atom        175   69    H     "Dimethyl Disulfide H3C"       1     1.008    1
atom        176   67    S     "Ethyl Sulfide S"             16    32.066    2
atom        177   68    H     "Ethyl Sulfide HS"             1     1.008    1
atom        178   62    C     "Ethyl Sulfide CH2"            6    12.011    4
atom        179   69    H     "Ethyl Sulfide H2C"            1     1.008    1
atom        180   70    C     "Ethyl Sulfide CH3"            6    12.011    4
atom        181   71    H     "Ethyl Sulfide H3C"            1     1.008    1
atom        182   67    S     "MeEt Sulfide S"              16    32.066    2
atom        183   62    C     "MeEt Sulfide CH3-S"           6    12.011    4
atom        184   69    H     "MeEt Sulfide H3C-S"           1     1.008    1
atom        185   62    C     "MeEt Sulfide CH2-S"           6    12.011    4
atom        186   69    H     "MeEt Sulfide CH2-S"           1     1.008    1
atom        187   70    C     "MeEt Sulfide CH3-C"           6    12.011    4
atom        188   71    H     "MeEt Sulfide H3C-C"           1     1.008    1
atom        189   72    S     "Dimethyl Sulfoxide S=O"      16    32.066    3
atom        190   73    O     "Dimethyl Sulfoxide S=O"       8    15.999    1
atom        191   74    C     "Dimethyl Sulfoxide CH3"       6    12.011    4
atom        192   75    H     "Dimethyl Sulfoxide H3C"       1     1.008    1
atom        193   76    S     "Methyl Sulfonate SO3-"       16    32.066    4
atom        194   77    O     "Methyl Sulfonate SO3-"        8    15.999    1
atom        195   40    C     "Methyl Sulfonate CH3"         6    12.011    4
atom        196   78    H     "Methyl Sulfonate H3C"         1     1.008    1
atom        197   76    S     "Ethyl Sulfonate SO3-"        16    32.066    4
atom        198   77    O     "Ethyl Sulfonate SO3-"         8    15.999    1
atom        199   40    C     "Ethyl Sulfonate CH2"          6    12.011    4
atom        200   78    H     "Ethyl Sulfonate H2C"          1     1.008    1
atom        201   40    C     "Ethyl Sulfonate CH3"          6    12.011    4
atom        202   41    H     "Ethyl Sulfonate H3C"          1     1.008    1
atom        203   40    C     "Propyl Sulfonate Me-CH2"      6    12.011    4
atom        204   42    H     "Propyl Sulfonate Me-CH2"      1     1.008    1
atom        205   40    C     "Propyl Sulfonate CH3"         6    12.011    4
atom        206   41    H     "Propyl Sulfonate H3C"         1     1.008    1
atom        207   79    C     "Hydrogen Cyanide CN"          6    12.011    2
atom        208   80    N     "Hydrogen Cyanide CN"          7    14.007    1
atom        209   81    H     "Hydrogen Cyanide HCN"         1     1.008    1
atom        210   79    C     "Acetonitrile CN"              6    12.011    2
atom        211   80    N     "Acetonitrile CN"              7    14.007    1
atom        212   82    C     "Acetonitrile CH3"             6    12.011    4
atom        213   83    H     "Acetonitrile H3C"             1     1.008    1
atom        214   84    C     "Tricyanomethide CN"           6    12.011    2
atom        215   85    N     "Tricyanomethide CN"           7    14.007    1
atom        216   86    C     "Tricyanomethide >C-"          6    12.011    3
atom        217   87    C     "Benzene C"                    6    12.011    3
atom        218   88    H     "Benzene HC"                   1     1.008    1
atom        219   89    C     "Ethylbenzene C1-CH2"          6    12.011    3
atom        220   89    C     "Ethylbenzene C2"              6    12.011    3
atom        221   89    C     "Ethylbenzene C3"              6    12.011    3
atom        222   89    C     "Ethylbenzene C4"              6    12.011    3
atom        223   90    H     "Ethylbenzene H2"              1     1.008    1
atom        224   90    H     "Ethylbenzene H3"              1     1.008    1
atom        225   90    H     "Ethylbenzene H4"              1     1.008    1
atom        226   40    C     "Ethylbenzene CH2"             6    12.011    4
atom        227   41    H     "Ethylbenzene H2C"             1     1.008    1
atom        228   40    C     "Ethylbenzene CH3"             6    12.011    4
atom        229   41    H     "Ethylbenzene H3C"             1     1.008    1
atom        230   36    O     "Phenol OH"                    8    15.999    2
atom        231   37    H     "Phenol HO"                    1     1.008    1
atom        232   89    C     "Phenol C1-OH"                 6    12.011    3
atom        233   89    C     "Phenol C2"                    6    12.011    3
atom        234   89    C     "Phenol C3"                    6    12.011    3
atom        235   89    C     "Phenol C4"                    6    12.011    3
atom        236   90    H     "Phenol H2"                    1     1.008    1
atom        237   90    H     "Phenol H3"                    1     1.008    1
atom        238   90    H     "Phenol H4"                    1     1.008    1
atom        239   40    C     "Toluene CH3"                  6    12.011    4
atom        240   41    H     "Toluene H3C"                  1     1.008    1
atom        241   89    C     "Toluene C1-CH3"               6    12.011    3
atom        242   89    C     "Toluene C2"                   6    12.011    3
atom        243   89    C     "Toluene C3"                   6    12.011    3
atom        244   89    C     "Toluene C4"                   6    12.011    3
atom        245   90    H     "Toluene H2"                   1     1.008    1
atom        246   90    H     "Toluene H3"                   1     1.008    1
atom        247   90    H     "Toluene H4"                   1     1.008    1
atom        248   36    O     "p-Cresol OH"                  8    15.999    2
atom        249   37    H     "p-Cresol HO"                  1     1.008    1
atom        250   40    C     "p-Cresol CH3"                 6    12.011    4
atom        251   51    H     "p-Cresol H3C"                 1     1.008    1
atom        252   89    C     "p-Cresol C1-CH3"              6    12.011    3
atom        253   89    C     "p-Cresol C2"                  6    12.011    3
atom        254   89    C     "p-Cresol C3"                  6    12.011    3
atom        255   89    C     "p-Cresol C4-OH"               6    12.011    3
atom        256   90    H     "p-Cresol H2"                  1     1.008    1
atom        257   90    H     "p-Cresol H3"                  1     1.008    1
atom        258   91    N     "Imidazole NH"                 7    14.007    3
atom        259   46    H     "Imidazole HN"                 1     1.008    1
atom        260   92    C     "Imidazole N-C-N"              6    12.011    3
atom        261   93    H     "Imidazole HC"                 1     1.008    1
atom        262   91    N     "Imidazole N=C-"               7    14.007    2
atom        263   92    C     "Imidazole C-N=C"              6    12.011    3
atom        264   93    H     "Imidazole HC"                 1     1.008    1
atom        265   92    C     "Imidazole C-NH-"              6    12.011    3
atom        266   93    H     "Imidazole HC"                 1     1.008    1
atom        267   40    C     "4-Ethylimidazole CH2"         6    12.011    4
atom        268   41    H     "4-Ethylimidazole H2C"         1     1.008    1
atom        269   40    C     "4-Ethylimidazole CH3"         6    12.011    4
atom        270   41    H     "4-Ethylimidazole H3C"         1     1.008    1
atom        271   91    N     "4-Ethylimidazole ND"          7    14.007    3
atom        272   46    H     "4-Ethylimidazole HND"         1     1.008    1
atom        273   92    C     "4-Ethylimidazole CE"          6    12.011    3
atom        274   93    H     "4-Ethylimidazole HCE"         1     1.008    1
atom        275   91    N     "4-Ethylimidazole NE"          7    14.007    3
atom        276   46    H     "4-Ethylimidazole HNE"         1     1.008    1
atom        277   92    C     "4-Ethylimidazole CD"          6    12.011    3
atom        278   93    H     "4-Ethylimidazole HCD"         1     1.008    1
atom        279   92    C     "4-Ethylimidazole CG"          6    12.011    3
atom        280   91    N     "Indole N"                     7    14.007    3
atom        281   46    H     "Indole HN"                    6     1.008    1
atom        282   94    C     "Indole C2"                    6    12.011    3
atom        283   94    C     "Indole C3"                    6    12.011    3
atom        284   94    C     "Indole C4"                    6    12.011    3
atom        285   94    C     "Indole C5"                    6    12.011    3
atom        286   94    C     "Indole C6"                    6    12.011    3
atom        287   94    C     "Indole C7"                    6    12.011    3
atom        288   94    C     "Indole C3a"                   6    12.011    3
atom        289   94    C     "Indole C7a"                   6    12.011    3
atom        290   88    H     "Indole HC2"                   1     1.008    1
atom        291   88    H     "Indole HC3"                   1     1.008    1
atom        292   88    H     "Indole HC4"                   1     1.008    1
atom        293   88    H     "Indole HC5"                   1     1.008    1
atom        294   88    H     "Indole HC6"                   1     1.008    1
atom        295   88    H     "Indole HC7"                   1     1.008    1
atom        296   91    N     "3-Ethylindole N"              7    14.007    3
atom        297   46    H     "3-Ethylindole HN"             1     1.008    1
atom        298   94    C     "3-Ethylindole C2"             6    12.011    3
atom        299   94    C     "3-Ethylindole C3"             6    12.011    3
atom        300   94    C     "3-Ethylindole C4"             6    12.011    3
atom        301   94    C     "3-Ethylindole C5"             6    12.011    3
atom        302   94    C     "3-Ethylindole C6"             6    12.011    3
atom        303   94    C     "3-Ethylindole C7"             6    12.011    3
atom        304   94    C     "3-Ethylindole C3a"            6    12.011    3
atom        305   94    C     "3-Ethylindole C7a"            6    12.011    3
atom        306   88    H     "3-Ethylindole HC2"            1     1.008    1
atom        307   88    H     "3-Ethylindole HC4"            1     1.008    1
atom        308   88    H     "3-Ethylindole HC5"            1     1.008    1
atom        309   88    H     "3-Ethylindole HC6"            1     1.008    1
atom        310   88    H     "3-Ethylindole HC7"            1     1.008    1
atom        311   95    C     "3-Ethylindole CH2"            6    12.011    4
atom        312   41    H     "3-Ethylindole H2C"            1     1.008    1
atom        313   95    C     "3-Ethylindole CH3"            6    12.011    4
atom        314   41    H     "3-Ethylindole H3C"            1     1.008    1
atom        315   91    N     "3-Formylindole N"             7    14.007    3
atom        316   46    H     "3-Formylindole HN"            1     1.008    1
atom        317   94    C     "3-Formylindole C2"            6    12.011    3
atom        318   94    C     "3-Formylindole C3"            6    12.011    3
atom        319   94    C     "3-Formylindole C4"            6    12.011    3
atom        320   94    C     "3-Formylindole C5"            6    12.011    3
atom        321   94    C     "3-Formylindole C6"            6    12.011    3
atom        322   94    C     "3-Formylindole C7"            6    12.011    3
atom        323   94    C     "3-Formylindole C3a"           6    12.011    3
atom        324   94    C     "3-Formylindole C7a"           6    12.011    3
atom        325   88    H     "3-Formylindole HC2"           1     1.008    1
atom        326   88    H     "3-Formylindole HC4"           1     1.008    1
atom        327   88    H     "3-Formylindole HC5"           1     1.008    1
atom        328   88    H     "3-Formylindole HC6"           1     1.008    1
atom        329   88    H     "3-Formylindole HC7"           1     1.008    1
atom        330   53    C     "3-Formylindole C=O"           6    12.011    3
atom        331   96    O     "3-Formylindole O=C"           8    15.999    1
atom        332   63    H     "3-Formylindole HC=O"          1     1.008    1
atom        333   97    N     "Benzamidine N"                7    14.007    3
atom        334   98    H     "Benzamidine HN"               1     1.008    1
atom        335   99    C     "Benzamidine N-C-N"            6    12.011    3
atom        336   89    C     "Benzamidine C1-CN2"           6    12.011    3
atom        337   89    C     "Benzamidine C2"               6    12.011    3
atom        338   89    C     "Benzamidine C3"               6    12.011    3
atom        339   89    C     "Benzamidine C4"               6    12.011    3
atom        340   90    H     "Benzamidine H2"               1     1.008    1
atom        341   90    H     "Benzamidine H3"               1     1.008    1
atom        342   90    H     "Benzamidine H4"               1     1.008    1
atom        343  100    N     "Pyridinium N"                 7    14.007    3
atom        344   87    C     "Pyridinium C2"                6    12.011    3
atom        345   87    C     "Pyridinium C3"                6    12.011    3
atom        346   87    C     "Pyridinium C4"                6    12.011    3
atom        347  101    H     "Pyridinium HN"                1     1.008    1
atom        348   88    H     "Pyridinium H2"                1     1.008    1
atom        349   88    H     "Pyridinium H3"                1     1.008    1
atom        350   88    H     "Pyridinium H4"                1     1.008    1


      ################################
      ##                            ##
      ##  Van der Waals Parameters  ##
      ##                            ##
      ################################


vdw           1               2.9900     0.0080
vdw           2               3.1500     0.0730
vdw           3               3.8200     0.2600
vdw           4               4.0900     0.3590
vdw           5               4.3700     0.4980
vdw           6               2.2000     0.0660
vdw           7               2.9550     0.2800
vdw           8               3.6800     0.3500
vdw           9               3.9000     0.3800
vdw          10               4.1400     0.4200
vdw          11               2.9400     0.3000
vdw          12               3.6300     0.3500
vdw          13               2.6800     0.2220
vdw          14               3.4300     0.2500
vdw          15               4.1200     0.3400
vdw          16               4.3200     0.4300
vdw          17               4.6100     0.5200
vdw          18               4.1200     0.1010
vdw          19               3.7100     0.1100
vdw          20               3.9000     0.1000
vdw          21               3.2200     0.1200
vdw          22               4.5300     0.3500
vdw          23               3.2200     0.1200
vdw          24               3.7100     0.0800      0.990
vdw          25               3.7800     0.1010
vdw          26               2.9000     0.0220      0.900
vdw          27               3.8200     0.1010
vdw          28               2.9600     0.0240      0.920
vdw          29               3.8200     0.1010
vdw          30               2.9800     0.0240      0.940
vdw          31               3.6500     0.1010
vdw          32               2.9400     0.0260      0.910
vdw          33               3.6000     0.1040
vdw          34               3.4050     0.1100
vdw          35               2.6550     0.0135      0.910
vdw          36               3.4050     0.1100
vdw          37               2.6550     0.0135      0.910
vdw          38               3.7600     0.1010
vdw          39               2.8700     0.0240      0.910
vdw          40               3.8200     0.1010
vdw          41               2.9600     0.0240      0.920
vdw          42               2.9800     0.0240      0.920
vdw          43               3.6500     0.1010
vdw          44               2.8900     0.0240      0.910
vdw          45               3.7100     0.1050
vdw          46               2.7000     0.0200      0.910
vdw          47               3.7100     0.1050
vdw          48               2.4800     0.0115      0.900
vdw          49               3.7100     0.1050
vdw          50               2.7000     0.0200      0.910
vdw          51               2.8800     0.0240      0.910
vdw          52               2.9600     0.0220      0.920
vdw          53               3.8200     0.1060
vdw          54               2.8000     0.0260      0.910
vdw          55               3.3000     0.1120
vdw          56               3.7100     0.1100
vdw          57               2.5900     0.0220      0.900
vdw          58               2.9500     0.0260      0.910
vdw          59               2.9300     0.0260      0.910
vdw          60               3.4050     0.1100
vdw          61               2.6550     0.0150      0.910
vdw          62               3.7800     0.1060
vdw          63               2.9200     0.0300      0.920
vdw          64               2.9800     0.0240      0.940
vdw          65               4.0050     0.3550
vdw          66               2.7700     0.0240      0.960
vdw          67               4.0050     0.3550
vdw          68               2.7700     0.0240      0.960
vdw          69               2.8700     0.0330      0.900
vdw          70               3.8200     0.1040
vdw          71               2.9800     0.0240      0.920
vdw          72               3.9100     0.3850
vdw          73               3.5200     0.1120
vdw          74               3.8200     0.1010
vdw          75               2.9100     0.0245      0.910
vdw          76               3.9100     0.3850
vdw          77               3.5100     0.1120
vdw          78               2.9100     0.0330      0.900
vdw          79               3.8200     0.1010
vdw          80               3.5800     0.1100
vdw          81               2.9200     0.0300      0.920
vdw          82               3.8200     0.1010
vdw          83               2.9100     0.0330      0.900
vdw          84               3.8200     0.1060
vdw          85               3.6900     0.1100
vdw          86               3.8200     0.1060
vdw          87               3.8000     0.0890
vdw          88               2.9800     0.0260      0.920
vdw          89               3.8000     0.0910
vdw          90               2.9800     0.0260      0.920
vdw          91               3.7100     0.1050
vdw          92               3.7800     0.1010
vdw          93               3.0000     0.0240      0.940
vdw          94               3.8000     0.1010
vdw          95               3.8200     0.1040
vdw          96               3.3000     0.1140
vdw          97               3.7100     0.1100
vdw          98               2.5900     0.0220      0.900
vdw          99               3.6500     0.1010
vdw         100               3.7100     0.1050
vdw         101               2.7000     0.0200      0.910


      #####################################
      ##                                 ##
      ##  Van der Waals Pair Parameters  ##
      ##                                 ##
      #####################################


vdwpr         8   15          4.2360     0.1512
vdwpr         8   16          4.3790     0.1664
vdwpr         8   17          4.6360     0.1720
vdwpr         9   15          4.3150     0.1859
vdwpr         9   16          4.4480     0.2068
vdwpr         9   17          4.6900     0.2145
vdwpr        10   15          4.3450     0.2894
vdwpr        10   16          4.4750     0.3307
vdwpr        10   17          4.7110     0.3466


      ##################################
      ##                              ##
      ##  Bond Stretching Parameters  ##
      ##                              ##
      ##################################


bond         18   19         1072.00     1.1874
bond         20   21          320.00     1.3900
bond         22   23          350.00     1.6260
bond         24   24         1613.00     1.0980
bond         25   26          395.50     1.0835
bond         27   27          323.00     1.5247
bond         27   28          341.00     1.1020
bond         27   29          323.00     1.5247
bond         27   31          323.00     1.5247
bond         29   29          323.00     1.5247
bond         29   30          341.00     1.1120
bond         31   32          341.00     1.1120
bond         33   89          453.20     1.4950
bond         34   35          556.85     0.9572
bond         36   37          615.90     0.9670
bond         36   38          500.10     1.4130
bond         36   40          465.10     1.4130
bond         36   43          410.10     1.4130
bond         36   89          676.60     1.3550
bond         38   39          385.00     1.1120
bond         39   40          385.00     1.1120
bond         39   43          341.00     1.1120
bond         40   40          453.00     1.5247
bond         40   41          385.00     1.1120
bond         40   42          385.00     1.1120
bond         40   43          323.00     1.5247
bond         40   44          378.00     1.1120
bond         40   49          381.30     1.4655
bond         40   51          385.00     1.0860
bond         40   52          341.00     1.1120
bond         40   53          345.30     1.5090
bond         40   56          374.80     1.4460
bond         40   58          341.00     1.1120
bond         40   59          341.00     1.1120
bond         40   62          345.30     1.5090
bond         40   64          341.00     1.1120
bond         40   76          220.00     1.8015
bond         40   78          341.00     1.1120
bond         40   89          453.20     1.4990
bond         40   92          453.20     1.4930
bond         41   95          341.00     1.1120
bond         45   46          516.50     1.0120
bond         46   91          461.90     1.0150
bond         46   91          467.60     1.0300
bond         47   48          461.90     1.0150
bond         49   50          515.90     1.0100
bond         53   54          356.40     1.1004
bond         53   55          601.80     1.2183
bond         53   56          482.00     1.3639
bond         53   63          314.40     1.0907
bond         53   94          345.30     1.5090
bond         53   96          705.00     1.2053
bond         55   62          705.00     1.2255
bond         56   57          542.00     1.0034
bond         60   61          514.40     0.9737
bond         60   62          431.60     1.3498
bond         62   63          314.40     1.0907
bond         62   67          235.80     1.8250
bond         62   69          395.50     1.0850
bond         62   70          323.00     1.5247
bond         65   66          278.40     1.3420
bond         67   68          315.40     1.3320
bond         70   71          341.00     1.1120
bond         72   73          494.20     1.5100
bond         72   74          197.20     1.8062
bond         74   75          388.20     1.0893
bond         76   77          570.00     1.4780
bond         79   80         1195.00     1.1700
bond         79   81          462.00     1.0650
bond         79   82          380.00     1.4570
bond         82   83          390.00     1.0873
bond         84   85         1110.00     1.1530
bond         84   86          420.00     1.4065
bond         87   87          471.90     1.3820
bond         87   88          370.50     1.0790
bond         87  100          653.90     1.3550
bond         88   94          370.50     1.1010
bond         89   89          471.90     1.3820
bond         89   90          370.50     1.0790
bond         89   99          323.00     1.5250
bond         91   92          653.90     1.3450
bond         91   94          653.90     1.3550
bond         92   92          539.60     1.3520
bond         92   93          370.50     1.0810
bond         94   94          471.90     1.3887
bond         94   95          345.30     1.5090
bond         95   95          323.00     1.5247
bond         97   98          487.00     1.0280
bond         97   99          491.40     1.3250
bond        100  101          467.60     1.0300


      ################################
      ##                            ##
      ##  Angle Bending Parameters  ##
      ##                            ##
      ################################


angle        21   20   21      65.00     109.47
anglef       23   22   23      40.00     180.00        4.0
angle        26   25   26      39.57     109.47
angle        27   27   28      42.44     109.80     109.31     110.70
angle        28   27   28      39.57     107.60     107.80     109.47
angle        28   27   29      42.44     109.80     109.31     110.70
angle        28   27   31      42.00     110.70
angle        27   29   27      48.20     109.50     110.20     111.00
angle        27   29   29      48.20     109.50     110.20     111.00
angle        29   29   29      48.20     109.50     110.20     111.00
angle        27   29   30      42.44     109.80     109.31     110.70
angle        29   29   30      42.44     109.80     109.31     110.70
angle        30   29   30      39.57     107.60     107.80     109.47
angle        27   31   27      48.20     109.50     110.20     111.00
angle        27   31   29      48.20     109.50     110.20     111.00
angle        29   31   29      48.20     109.50     110.20     111.00
angle        27   31   32      42.00     112.80
angle        29   31   32      42.00     112.80
angle        35   34   35      48.70     108.50
angle        37   36   38      64.96     106.80
angle        37   36   40      64.96     106.80
angle        37   36   43      53.96     106.80
angle        37   36   89      25.90     109.00
angle        40   36   40      88.50     106.00
angle        36   38   39      60.99     110.00     108.90     108.70
angle        39   38   39      45.57     107.60     107.80     109.47
angle        36   40   39      60.99     110.00     108.90     108.70
angle        36   40   40      65.71     107.50     107.00     107.90
angle        36   40   44      70.00     110.00     108.90     108.70
angle        39   40   39      40.57     107.60     107.80     109.47
angle        39   40   40      45.44     109.80     109.31     110.70
angle        40   40   40      48.20     109.50     110.20     111.00
angle        40   40   41      45.44     109.80     109.31     110.70
angle        40   40   42      45.44     109.80     109.31     110.70
angle        40   40   49      56.11     109.47     108.00     111.00
angle        40   40   51      42.44     109.80     109.31     110.70
angle        40   40   52      42.44     109.80     109.31     110.70
angle        40   40   56      53.96     109.48     111.30     111.80
angle        40   40   59      42.44     109.80     109.31     110.70
angle        40   40   76      53.00     108.00
angle        40   40   78      42.44     109.80     109.30     110.70
angle        40   40   89      38.85     110.60
angle        40   40   92      38.85     112.70
angle        41   40   41      40.57     107.60     107.80     109.47
angle        41   40   43      42.44     109.80     109.31     110.70
angle        41   40   89      39.57     109.50     109.31     110.40
angle        41   40   92      39.57     109.50
angle        42   40   42      45.57     107.60     107.80     109.47
angle        42   40   49      58.99     109.30
angle        44   40   44      43.57     107.60     107.80     109.47
angle        49   40   51      70.99     109.30
angle        49   40   52      58.99     109.30
angle        51   40   51      40.57     107.60     107.80     109.47
angle        51   40   89      39.57     109.50     109.31     110.40
angle        52   40   52      39.57     107.60     107.80     109.47
angle        53   40   58      38.85     109.49
angle        56   40   59      54.67     111.00
angle        58   40   58      39.57     107.60     107.80     109.47
angle        59   40   59      39.57     107.60     107.80     109.47
angle        62   40   64      38.85     109.49
angle        64   40   64      39.57     107.60     107.80     109.47
angle        76   40   78      35.00     108.15
angle        78   40   78      39.00     110.75
angle        36   43   39      58.99     110.00     108.90     108.70
angle        36   43   40      59.71     107.50     107.00     107.90
angle        39   43   40      42.44     109.80     109.31     110.70
angle        40   43   40      48.20     109.50     110.20     111.00
angle        46   45   46      43.52     106.40     106.80       0.00
angle        48   47   48      43.52     109.47
angle        40   49   40      51.80     107.20     108.20       0.00
angle        40   49   50      43.16     108.10     110.90       0.00
angle        50   49   50      34.50     106.40     107.10       0.00
angle        40   53   55      80.00     122.40
angle        40   53   56      70.00     113.40
angle        54   53   55      68.34     119.20
angle        54   53   56      31.65     109.30
angle        55   53   56      76.98     124.20
angle        63   53   94      33.38     116.10     117.30       0.00
angle        63   53   96      61.15     119.20     119.20       0.00
angle        94   53   96      61.15     123.50     123.50       0.00
angle        40   56   40      54.67     122.50
angle        40   56   53      50.00     122.00
angle        40   56   57      32.01     117.00
angle        53   56   57      50.00     121.00
angle        57   56   57      22.30     122.00
angle        61   60   62      49.64     108.70
angle        40   62   55      61.15     123.50     123.50       0.00
angle        40   62   60     111.51     110.30
angle        40   62   63      33.38     116.10     117.30       0.00
angle        55   62   60     122.30     121.50     122.50       0.00
angle        55   62   63      61.15     119.20     119.20       0.00
angle        55   62   63      61.15     119.20     119.20       0.00
angle        55   62   63      61.15     119.20     119.20       0.00
angle        55   62   63      61.15     119.20     119.20       0.00
angle        60   62   63      39.57     107.00
angle        60   62   63      39.57     107.00
angle        60   62   63      39.57     107.00
angle        60   62   63      39.57     107.00
angle        63   62   63      46.76     115.50
angle        67   62   69      60.24     110.80     110.80     108.00
angle        67   62   70      53.24     108.00     109.50     110.10
angle        69   62   69      39.57     107.60     107.80     109.47
angle        69   62   70      42.44     109.80     109.31     110.70
angle        66   65   66      52.52      92.90
angle        62   67   62      60.43      95.90
angle        62   67   68      46.76      96.00
angle        62   70   71      42.44     109.80     109.31     110.70
angle        71   70   71      39.57     107.60     107.80     109.47
angle        73   72   74      57.25     106.88
angle        74   72   74      87.39      92.98
angle        72   74   75      44.07     107.06
angle        75   74   75      38.95     110.07
angle        40   76   77      60.00     104.60
angle        77   76   77      70.00     113.50
angle        80   79   81      18.40     180.00
angle        80   79   82      65.00     180.00
angle        79   82   83      45.00     109.92
angle        83   82   83      40.00     109.02
angle        85   84   86      42.00     180.00
angle        84   86   84      65.00     120.00
angle        87   87   87      63.31     120.00
angle        87   87   88      35.25     120.00     120.50       0.00
angle        87   87  100      69.78     121.00
angle        88   87  100      38.13     119.00
angle        36   89   89      43.16     120.00
angle        40   89   89      33.81     122.30
angle        89   89   89      64.67     121.70
angle        89   89   90      35.25     120.00     120.50       0.00
angle        89   89   99      54.67     121.70
angle        46   91   92      35.25     125.50
angle        46   91   94      35.25     125.50
angle        92   91   92      86.33     105.10     107.10       0.00
angle        92   91   92      86.33     105.10     107.10       0.00
angle        94   91   94      86.33     109.00
angle        40   92   91      35.97     122.00
angle        40   92   92      35.97     131.00
angle        91   92   91      28.78     112.10
angle        91   92   92      47.48     107.00
angle        91   92   93      38.13     122.50
angle        92   92   93      35.25     128.00
angle        53   94   94      54.67     121.70
angle        88   94   91      38.13     122.50
angle        88   94   94      35.25     128.00
angle        91   94   94      47.48     109.00
angle        94   94   94      63.31     120.00
angle        94   94   95      61.87     127.00
angle        41   95   41      39.57     107.60     107.80     109.47
angle        41   95   94      61.15     109.80     109.31     110.70
angle        41   95   95      42.44     109.80     109.31     110.70
angle        94   95   95      48.20     109.50     110.20     111.00
angle        98   97   98      29.50     123.00
angle        98   97   99      41.70     120.50
angle        89   99   97      28.80     120.00
angle        97   99   97      28.80     120.00
angle        87  100   87      86.33     112.60
angle        87  100  101      35.25     123.70


      ################################
      ##                            ##
      ##   Stretch-Bend Parameters  ##
      ##                            ##
      ################################


strbnd       27   27   28      11.50      11.50
strbnd       28   27   29      11.50      11.50
strbnd       28   27   31      11.50      11.50
strbnd       27   29   27      18.70      18.70
strbnd       27   29   29      18.70      18.70
strbnd       29   29   29      18.70      18.70
strbnd       27   29   30      11.50      11.50
strbnd       29   29   30      11.50      11.50
strbnd       27   31   27      18.70      18.70
strbnd       27   31   29      18.70      18.70
strbnd       29   31   29      18.70      18.70
strbnd       27   31   32      11.50      11.50
strbnd       29   31   32      11.50      11.50
strbnd       37   36   40      -4.50      38.00
strbnd       37   36   89      12.95      12.95
strbnd       38   36   37      38.00      -4.50
strbnd       40   36   40      38.00      38.00
strbnd       39   38   36      -4.50      38.00
strbnd       36   40   40      38.00      38.00
strbnd       36   40   41      11.50      11.50
strbnd       36   40   44      38.00      -4.50
strbnd       39   40   36      -4.50      38.00
strbnd       39   40   40      -4.50      38.00
strbnd       40   40   40      18.70      18.70
strbnd       40   40   41      38.00      -4.50
strbnd       40   40   42      38.00      -4.50
strbnd       40   40   49      18.70      18.70
strbnd       40   40   51      11.50      11.50
strbnd       40   40   52      11.50      11.50
strbnd       40   40   53      18.70      18.70
strbnd       40   40   56      18.70      18.70
strbnd       40   40   58      11.50      11.50
strbnd       40   40   59      11.50      11.50
strbnd       40   40   76      18.70      18.70
strbnd       40   40   78      11.50      11.50
strbnd       40   40   89      18.70      18.70
strbnd       40   40   92      18.70      18.70
strbnd       41   40   43      11.50      11.50
strbnd       41   40   49      11.50      11.50
strbnd       41   40   89      11.50      11.50
strbnd       41   40   92      11.50      11.50
strbnd       42   40   49      11.50      11.50
strbnd       42   40   53      11.50      11.50
strbnd       49   40   52      11.50      11.50
strbnd       51   40   49      11.50      11.50
strbnd       51   40   89      11.50      11.50
strbnd       53   40   58      11.50      11.50
strbnd       56   40   59      11.50      11.50
strbnd       62   40   64      11.50      11.50
strbnd       78   40   76      11.50      11.50
strbnd       36   43   40      38.00      38.00
strbnd       39   43   36      -4.50      38.00
strbnd       39   43   40      -4.50      38.00
strbnd       40   43   40      38.00      38.00
strbnd       40   49   40       7.20       7.20
strbnd       40   49    0       4.30       4.30
strbnd       40   49   50       4.30       4.30
strbnd       40   53   55      18.70      18.70
strbnd       40   53   56      18.70      18.70
strbnd       54   53   55      11.50      11.50
strbnd       54   53   56      11.50      11.50
strbnd       55   53   56      18.70      18.70
strbnd       63   53   94      11.50      11.50
strbnd       63   53   96      11.50      11.50
strbnd       94   53   96      18.70      18.70
strbnd       40   56   40       7.20       7.20
strbnd       40   56   53       7.20       7.20
strbnd       40   56   57       4.30       4.30
strbnd       53   56   57       4.30       4.30
strbnd       61   60   62      12.95      12.95
strbnd       40   62   55      18.70      18.70
strbnd       40   62   60      18.70      18.70
strbnd       40   62   63      11.50      11.50
strbnd       55   62   60      18.70      18.70
strbnd       55   62   63      11.50      11.50
strbnd       60   62   63      11.50      11.50
strbnd       67   62   70      18.70      18.70
strbnd       69   62   67      11.50      11.50
strbnd       69   62   70      11.50      11.50
strbnd       62   67   62      -5.75      -5.75
strbnd       62   67   67      -5.75      -5.75
strbnd       62   67   68       1.45       1.45
strbnd       62   70   71      11.50      11.50
strbnd       73   72   74      -5.75      -5.75
strbnd       74   72   74      -5.75      -5.75
strbnd       72   74   75      11.50      11.50
strbnd       80   79   82      18.70      18.70
strbnd       79   82   83      11.50      11.50
strbnd       86   84   85      18.70      18.70
strbnd       84   86   84      18.70      18.70
strbnd       87   87   87      18.70      18.70
strbnd       87   87   88      11.50      11.50
strbnd       87   87  100      18.70      18.70
strbnd       88   87  100      11.50      11.50
strbnd       36   89   89      18.70      18.70
strbnd       40   89   89      18.70      18.70
strbnd       89   89   89      18.70      18.70
strbnd       89   89   90      38.00      11.60
strbnd       89   89   99      18.70      18.70
strbnd       46   91   92       4.30       4.30
strbnd       46   91   94       4.30       4.30
strbnd       46   91   95       4.30       4.30
strbnd        0   91   95       4.30       4.30
strbnd       92   91   92      14.40      14.40
strbnd       94   91   94      14.40      14.40
strbnd       40   92   91      18.70      18.70
strbnd       40   92   92      18.70      18.70
strbnd       91   92   91      18.70      18.70
strbnd       91   92   92      18.70      18.70
strbnd       91   92   93      11.50      11.50
strbnd       92   92   93      11.50      11.50
strbnd       53   94   94      18.70      18.70
strbnd       88   94   91      11.50      11.50
strbnd       88   94   94      11.50      11.50
strbnd       91   94   94      18.70      18.70
strbnd       94   94   94      18.70      18.70
strbnd       94   94   95      18.70      18.70


      ################################
      ##                            ##
      ##   Urey-Bradley Parameters  ##
      ##                            ##
      ################################


ureybrad     35   34   35      -7.60     1.5337


      ####################################
      ##                                ##
      ##  Out-of-Plane Bend Parameters  ##
      ##                                ##
      ####################################


opbend       40   53    0    0            42.40
opbend       54   53    0    0           140.30
opbend       55   53    0    0            46.80
opbend       56   53    0    0           107.90
opbend       63   53    0    0           140.30
opbend       94   53    0    0            42.40
opbend       96   53    0    0            46.80
opbend       40   56    0    0            12.90
opbend       53   56    0    0            12.90
opbend       57   56    0    0             5.80
opbend       40   62    0    0            42.40
opbend       55   62    0    0            46.80
opbend       60   62    0    0           107.90
opbend       63   62    0    0           140.30
opbend       84   86    0    0            14.40
opbend       87   87    0    0            14.40
opbend       88   87    0    0            15.10
opbend      100   87    0    0            18.00
opbend       33   89    0    0            14.40
opbend       36   89    0    0            14.40
opbend       40   89    0    0            14.40
opbend       89   89    0    0            14.40
opbend       90   89    0    0            15.10
opbend       99   89    0    0             7.20
opbend       53   94    0    0            14.40
opbend       88   94    0    0             7.90
opbend       91   94    0    0            14.40
opbend       94   94    0    0            14.40
opbend       95   94    0    0            14.40
opbend       98   97    0    0            12.90
opbend       99   97    0    0             3.60
opbend       89   99    0    0             1.40
opbend       97   99    0    0             1.40
opbend       87  100    0    0            10.80
opbend      101  100    0    0            10.80


      ############################
      ##                        ##
      ##  Torsional Parameters  ##
      ##                        ##
      ############################


torsion      39    0   49   50      0.000 0.0 1  -0.014 180.0 2   0.295 0.0 3
torsion      28   27   27   28      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      28   27   29   27      0.000 0.0 1   0.000 180.0 2   0.341 0.0 3
torsion      28   27   29   29      0.000 0.0 1   0.000 180.0 2   0.341 0.0 3
torsion      28   27   29   30      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      27   29   29   27      0.854 0.0 1  -0.374 180.0 2   0.108 0.0 3
torsion      27   29   29   29      0.854 0.0 1  -0.374 180.0 2   0.108 0.0 3
torsion      27   29   29   30      0.000 0.0 1   0.000 180.0 2   0.341 0.0 3
torsion      29   29   29   29      0.854 0.0 1  -0.374 180.0 2   0.108 0.0 3
torsion      29   29   29   30      0.000 0.0 1   0.000 180.0 2   0.108 0.0 3
torsion      30   29   29   30      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      37   36   38   39      0.000 0.0 1   0.000 180.0 2   0.274 0.0 3
torsion      37   36   40   39      0.000 0.0 1   0.000 180.0 2   0.274 0.0 3
torsion      37   36   40   40     -1.447 0.0 1   0.531 180.0 2   0.317 0.0 3
torsion      40   36   40   44      0.000 0.0 1   0.000 180.0 2   0.597 0.0 3
torsion      37   36   43   39      0.000 0.0 1   0.000 180.0 2   0.266 0.0 3
torsion      37   36   43   40     -1.372 0.0 1   0.232 180.0 2   0.400 0.0 3
torsion      37   36   89   89      0.000 0.0 1   2.081 180.0 2   0.000 0.0 3
torsion      36   40   40   40     -1.150 0.0 1   0.000 180.0 2   1.280 0.0 3
torsion      36   40   40   41      0.000 0.0 1   0.000 180.0 2   0.300 0.0 3
torsion      36   40   40   42      0.000 0.0 1   0.000 180.0 2   0.300 0.0 3
torsion      39   40   40   40      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      39   40   40   41      0.000 0.0 1   0.000 180.0 2   0.424 0.0 3
torsion      39   40   40   42      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      40   40   40   40      0.185 0.0 1   0.170 180.0 2   0.520 0.0 3
torsion      40   40   40   41      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      40   40   40   49     -0.302 0.0 1   0.696 180.0 2   0.499 0.0 3
torsion      40   40   40   51      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      40   40   40   52      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      40   40   40   76      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      40   40   40   78      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      41   40   40   41      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   40   42      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      41   40   40   49      0.000 0.0 1   0.000 180.0 2   0.374 0.0 3
torsion      41   40   40   51      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   40   56      0.000 0.0 1   0.000 180.0 2   0.500 0.0 3
torsion      41   40   40   59      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   40   76      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      41   40   40   78      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   40   89      0.000 0.0 1   0.000 180.0 2   0.500 0.0 3
torsion      41   40   40   92      0.000 0.0 1   0.000 180.0 2   0.500 0.0 3
torsion      42   40   40   49      0.000 0.0 1   0.000 180.0 2   0.374 0.0 3
torsion      42   40   40   51      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      42   40   40   76      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      42   40   40   78      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      49   40   40   52      0.000 0.0 1   0.000 180.0 2   0.374 0.0 3
torsion      52   40   40   52      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   43   36      0.000 0.0 1   0.000 180.0 2   0.300 0.0 3
torsion      41   40   43   39      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      41   40   43   40      0.000 0.0 1   0.000 180.0 2   0.280 0.0 3
torsion      40   40   49   40      0.958 0.0 1  -0.155 180.0 2   0.766 0.0 3
torsion      40   40   49   50     -0.107 0.0 1   0.512 180.0 2   0.365 0.0 3
torsion      42   40   49   40      0.072 0.0 1  -0.012 180.0 2   0.563 0.0 3
torsion      51   40   49   40      0.072 0.0 1  -0.012 180.0 2   0.563 0.0 3
torsion      51   40   49   50      0.000 0.0 1   0.661 180.0 2   0.288 0.0 3
torsion      52   40   49   40      0.121 0.0 1  -0.648 180.0 2   0.199 0.0 3
torsion      52   40   49   50      0.121 0.0 1  -0.648 180.0 2   0.199 0.0 3
torsion      58   40   53   55      0.000 0.0 1   0.000 180.0 2   0.235 0.0 3
torsion      58   40   53   56      0.000 0.0 1   0.000 180.0 2  -0.010 0.0 3
torsion      40   40   56   53      0.660 0.0 1  -0.456 180.0 2   0.254 0.0 3
torsion      40   40   56   57     -0.660 0.0 1  -0.420 180.0 2  -0.254 0.0 3
torsion      59   40   56   40      0.000 0.0 1   0.000 180.0 2   0.460 0.0 3
torsion      59   40   56   53      0.000 0.0 1   0.000 180.0 2  -0.126 0.0 3
torsion      59   40   56   57      0.000 0.0 1   0.000 180.0 2   0.000 0.0 3
torsion      64   40   62   55     -0.154 0.0 1   0.044 180.0 2  -0.086 0.0 3
torsion      64   40   62   60      0.250 0.0 1   0.850 180.0 2   0.000 0.0 3
torsion      64   40   62   63      0.115 0.0 1   0.027 180.0 2   0.285 0.0 3
torsion      40   40   76   77      0.000 0.0 1   0.000 180.0 2   0.000 0.0 3
torsion      78   40   76   77      0.000 0.0 1   0.000 180.0 2   0.000 0.0 3
torsion      40   40   89   89     -0.800 0.0 1  -0.100 180.0 2  -0.550 0.0 3
torsion      41   40   89   89      0.000 0.0 1   0.000 180.0 2  -0.090 0.0 3
torsion      51   40   89   89      0.000 0.0 1   0.000 180.0 2  -0.090 0.0 3
torsion      40   40   92   91     -0.800 0.0 1  -0.100 180.0 2  -0.550 0.0 3
torsion      40   40   92   92     -0.800 0.0 1  -0.100 180.0 2  -0.550 0.0 3
torsion      41   40   92   91      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   40   92   92      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      40   53   56   40     -1.000 0.0 1   2.000 180.0 2   2.050 0.0 3
torsion      40   53   56   57      0.000 0.0 1   1.200 180.0 2   0.800 0.0 3
torsion      54   53   56   40      0.000 0.0 1   2.250 180.0 2   0.000 0.0 3
torsion      54   53   56   57      0.000 0.0 1   0.500 180.0 2   0.350 0.0 3
torsion      55   53   56   40      1.000 0.0 1   2.250 180.0 2  -2.250 0.0 3
torsion      55   53   56   57      0.000 0.0 1  -0.664 180.0 2  -0.357 0.0 3
torsion      63   53   94   94     -0.300 0.0 1   8.000 180.0 2   0.000 0.0 3
torsion      96   53   94   94      0.000 0.0 1   8.000 180.0 2   0.000 0.0 3
torsion      61   60   62   40      0.000 0.0 1   5.390 180.0 2   1.230 0.0 3
torsion      61   60   62   55     -1.200 0.0 1   5.390 180.0 2   0.400 0.0 3
torsion      61   60   62   63     -0.300 0.0 1   5.390 180.0 2   0.000 0.0 3
torsion      69   62   67   62      0.000 0.0 1   0.000 180.0 2   0.660 0.0 3
torsion      69   62   67   68      0.000 0.0 1   0.000 180.0 2   0.383 0.0 3
torsion      70   62   67   62     -0.440 0.0 1  -0.260 180.0 2   0.600 0.0 3
torsion      70   62   67   68     -1.096 0.0 1   0.079 180.0 2   0.384 0.0 3
torsion      67   62   70   71      0.000 0.0 1   0.000 180.0 2   0.475 0.0 3
torsion      69   62   70   71      0.000 0.0 1   0.000 180.0 2   0.238 0.0 3
torsion      73   72   74   75      0.000 0.0 1   0.000 180.0 2   0.500 0.0 3
torsion      74   72   74   75      0.000 0.0 1   0.000 180.0 2   0.500 0.0 3
torsion      80   79   82   83      0.000 0.0 1   0.000 180.0 2   0.000 0.0 3
torsion      85   84   86   84      0.000 0.0 1   0.000 180.0 2   0.000 0.0 3
torsion      87   87   87   87     -0.670 0.0 1   4.004 180.0 2   0.000 0.0 3
torsion      87   87   87   88      0.550 0.0 1   4.534 180.0 2  -0.550 0.0 3
torsion      87   87   87  100      0.000 0.0 1   5.470 180.0 2   0.000 0.0 3
torsion      88   87   87   88      0.000 0.0 1   4.072 180.0 2   0.000 0.0 3
torsion      88   87   87  100     -3.150 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      87   87  100   87      0.000 0.0 1  14.000 180.0 2   0.000 0.0 3
torsion      87   87  100  101     -3.150 0.0 1   8.000 180.0 2   0.000 0.0 3
torsion      88   87  100   87     -6.650 0.0 1  20.000 180.0 2   0.000 0.0 3
torsion      88   87  100  101     -0.530 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      36   89   89   89      0.000 0.0 1   4.470 180.0 2   0.000 0.0 3
torsion      36   89   89   90      0.000 0.0 1   4.470 180.0 2   0.000 0.0 3
torsion      40   89   89   89     -0.610 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      40   89   89   90      0.000 0.0 1   6.104 180.0 2   0.000 0.0 3
torsion      89   89   89   89     -0.670 0.0 1   4.004 180.0 2   0.000 0.0 3
torsion      89   89   89   90      0.550 0.0 1   4.534 180.0 2  -0.550 0.0 3
torsion      89   89   89   99     -0.610 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      90   89   89   90      0.000 0.0 1   4.072 180.0 2   0.000 0.0 3
torsion      90   89   89   99      0.000 0.0 1   6.104 180.0 2   0.000 0.0 3
torsion      89   89   99   97      0.000 0.0 1   2.304 180.0 2   0.000 0.0 3
torsion      46   91   92   40      0.000 0.0 1   4.104 180.0 2   0.000 0.0 3
torsion      46   91   92   91     -2.744 0.0 1  15.000 180.0 2   0.000 0.0 3
torsion      46   91   92   92     -3.150 0.0 1   8.000 180.0 2   0.000 0.0 3
torsion      46   91   92   93     -0.530 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      92   91   92   40      0.000 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      92   91   92   91      0.000 0.0 1  15.000 180.0 2   0.000 0.0 3
torsion      92   91   92   92      0.000 0.0 1  14.000 180.0 2   0.000 0.0 3
torsion      92   91   92   93      0.000 0.0 1   7.000 180.0 2   0.000 0.0 3
torsion      46   91   94   88     -0.530 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      46   91   94   94     -3.150 0.0 1   8.000 180.0 2   0.000 0.0 3
torsion      94   91   94   88     -6.650 0.0 1  20.000 180.0 2   0.000 0.0 3
torsion      94   91   94   94      0.000 0.0 1  14.000 180.0 2   0.000 0.0 3
torsion      40   92   92   91      0.000 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      40   92   92   93      0.000 0.0 1   4.102 180.0 2   0.000 0.0 3
torsion      91   92   92   91      0.900 0.0 1  15.000 180.0 2   0.000 0.0 3
torsion      91   92   92   93     -3.150 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      93   92   92   93      0.000 0.0 1  11.500 180.0 2   0.000 0.0 3
torsion      53   94   94   88      0.000 0.0 1   6.104 180.0 2   0.000 0.0 3
torsion      53   94   94   91      0.000 0.0 1   5.470 180.0 2   0.000 0.0 3
torsion      53   94   94   94     -0.610 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      88   94   94   88      0.000 0.0 1   7.072 180.0 2   0.000 0.0 3
torsion      88   94   94   91     -3.150 0.0 1   3.000 180.0 2   0.000 0.0 3
torsion      88   94   94   94      0.250 0.0 1   5.534 180.0 2  -0.550 0.0 3
torsion      88   94   94   95      0.000 0.0 1   6.104 180.0 2   0.000 0.0 3
torsion      91   94   94   94      0.000 0.0 1   5.470 180.0 2   0.000 0.0 3
torsion      91   94   94   95      0.000 0.0 1   5.470 180.0 2   0.000 0.0 3
torsion      94   94   94   94     -0.670 0.0 1   4.304 180.0 2   0.000 0.0 3
torsion      94   94   94   95     -0.610 0.0 1   4.212 180.0 2   0.000 0.0 3
torsion      94   94   95   41      0.000 0.0 1   0.000 180.0 2   0.341 0.0 3
torsion      94   94   95   95      0.260 0.0 1  -0.255 180.0 2   0.260 0.0 3
torsion      41   95   95   41      0.000 0.0 1   0.000 180.0 2   0.299 0.0 3
torsion      41   95   95   94      0.000 0.0 1   0.000 180.0 2   0.341 0.0 3
torsion      98   97   99   89      0.000 0.0 1   4.000 180.0 2   0.000 0.0 3
torsion      98   97   99   97      0.000 0.0 1   4.000 180.0 2   0.000 0.0 3


      #############################
      ##                         ##
      ##  Pi-Torsion Parameters  ##
      ##                         ##
      #############################


pitors       53   56            6.85
pitors       87   87            6.85
pitors       89   89            6.85


      ###################################
      ##                               ##
      ##  Atomic Multipole Parameters  ##
      ##                               ##
      ###################################


multipole     1    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     2    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     3    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     4    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     5    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     6    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     7    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     8    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole     9    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    10    0    0               1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    11    0    0               2.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    12    0    0               2.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    13    0    0               2.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    14    0    0              -1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    15    0    0              -1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    16    0    0              -1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    17    0    0              -1.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
#multipole    18   19    0              -0.50825
                                        0.00000    0.00000    0.83201
                                        0.03357
                                        0.00000    0.03357
                                        0.00000    0.00000   -0.06714
#multipole    19   18    0              -0.49175
                                        0.00000    0.00000    0.62101
                                       -0.46088
                                        0.00000   -0.46088
                                        0.00000    0.00000    0.92176
#multipole    18   19    0              -0.52726
                                        0.00000    0.00000    0.77131
                                        0.10528
                                        0.00000    0.10528
                                        0.00000    0.00000   -0.21056
#multipole    19   18    0              -0.47274
                                        0.00000    0.00000    0.63125
                                       -0.46405
                                        0.00000   -0.46405
                                        0.00000    0.00000    0.92810
#multipole    18   19    0              -0.52726
                                        0.00000    0.00000    0.77131
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
#multipole    19   18    0              -0.47274
                                        0.00000    0.00000    0.63125
                                       -0.24000
                                        0.00000   -0.24000
                                        0.00000    0.00000    0.48000
multipole    18   19    0              -0.50825
                                        0.00000    0.00000    0.77400
                                       -0.03000
                                        0.00000   -0.03000
                                        0.00000    0.00000    0.06000
multipole    19   18    0              -0.49175
                                        0.00000    0.00000    0.59100
                                       -0.22500
                                        0.00000   -0.22500
                                        0.00000    0.00000    0.45000
multipole    20    0    0               0.50724
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    21   20    0              -0.37681
                                        0.00000    0.00000    0.35197
                                       -0.33840
                                        0.00000   -0.33840
                                        0.00000    0.00000    0.67680
multipole    22    0    0               2.24240
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    23   22    0              -0.54040
                                        0.00000    0.00000   -0.00975
                                       -0.21692
                                        0.00000   -0.21692
                                        0.00000    0.00000    0.43384
multipole    24   24    0               0.00000
                                        0.00000    0.00000    0.12578
                                        0.16329
                                        0.00000    0.16329
                                        0.00000    0.00000   -0.32658
multipole    25   26   26              -0.22648
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    26   25   26               0.05662
                                        0.00000    0.00000   -0.12142
                                       -0.00797
                                        0.00000   -0.00797
                                        0.00000    0.00000    0.01594
multipole    27   27   28              -0.17655
                                        0.00000    0.00000    0.04700
                                       -0.22275
                                        0.00000   -0.22275
                                        0.00000    0.00000    0.44550
multipole    28   27   27               0.05885
                                        0.03092    0.00000   -0.07878
                                       -0.00954
                                        0.00000   -0.03313
                                        0.00313    0.00000    0.04267
multipole    29   31   30              -0.16638
                                        0.00000    0.00000    0.25069
                                       -0.24574
                                        0.00000   -0.24574
                                        0.00000    0.00000    0.49148
multipole    29   33   30              -0.16638
                                        0.00000    0.00000    0.25069
                                       -0.24574
                                        0.00000   -0.24574
                                        0.00000    0.00000    0.49148
multipole    30   29   31               0.05546
                                        0.01102    0.00000   -0.10399
                                        0.00247
                                        0.00000   -0.01906
                                        0.00852    0.00000    0.01659
multipole    30   29   33               0.05546
                                        0.01102    0.00000   -0.10399
                                        0.00247
                                        0.00000   -0.01906
                                        0.00852    0.00000    0.01659
multipole    31   31   29              -0.12028
                                        0.33700    0.00000    0.22666
                                        0.34191
                                        0.00000   -0.72407
                                       -0.04985    0.00000    0.38216
multipole    31  -31  -31              -0.12028
                                        0.00000    0.00000    0.27522
                                        0.47728
                                        0.00000   -0.35302
                                        0.00000    0.00000   -0.12426
multipole    32   31   31               0.06014
                                        0.00606    0.00000   -0.08554
                                       -0.00370
                                        0.00000   -0.01833
                                        0.00546    0.00000    0.02203
multipole    33   34   29              -0.05646
                                        0.00000    0.00000   -0.19753
                                       -0.37464
                                        0.00000    0.21807
                                        0.19091    0.00000    0.15657
multipole    33   34   31              -0.05646
                                        0.00000    0.00000   -0.19753
                                       -0.37464
                                        0.00000    0.21807
                                        0.19091    0.00000    0.15657
multipole    33   34   33              -0.05646
                                        0.00000    0.00000   -0.19753
                                       -0.37464
                                        0.00000    0.21807
                                        0.19091    0.00000    0.15657
multipole    34   33   29               0.05646
                                       -0.00037    0.00000   -0.03746
                                       -0.01137
                                        0.00000   -0.01137
                                        0.00000    0.00000    0.02274
multipole    34   33   31               0.05646
                                       -0.00037    0.00000   -0.03746
                                       -0.01137
                                        0.00000   -0.01137
                                        0.00000    0.00000    0.02274
multipole    34   33   33               0.05646
                                       -0.00037    0.00000   -0.03746
                                       -0.01137
                                        0.00000   -0.01137
                                        0.00000    0.00000    0.02274
multipole    35    0    0               0.00000
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    36  -37  -37              -0.51966
                                        0.00000    0.00000    0.14279
                                        0.37928
                                        0.00000   -0.41809
                                        0.00000    0.00000    0.03881
multipole    37   36   37               0.25983
                                       -0.03859    0.00000   -0.05818
                                       -0.03673
                                        0.00000   -0.10739
                                       -0.00203    0.00000    0.14412
multipole    38   39   40              -0.53771
                                        0.23239    0.00000    0.14256
                                        0.21444
                                        0.00000   -0.55826
                                       -0.21095    0.00000    0.34382
multipole    39   38   40               0.24154
                                       -0.04246    0.00000    0.00000
                                       -0.06223
                                        0.00000   -0.12152
                                       -0.02736    0.00000    0.18375
multipole    40   38   39               0.17575
                                        0.00000    0.00000    0.57127
                                       -0.44733
                                        0.00000   -0.44733
                                        0.00000    0.00000    0.89466
multipole    41   40   38               0.04014
                                        0.00000    0.00000   -0.10668
                                       -0.00544
                                        0.00000   -0.02734
                                       -0.01699    0.00000    0.03278
multipole    42   43   44              -0.50793
                                        0.30397    0.00000    0.07443
                                        0.21785
                                        0.00000   -0.54387
                                       -0.22371    0.00000    0.32602
multipole    43   42   44               0.24956
                                       -0.04674    0.00000    0.00000
                                       -0.04536
                                        0.00000   -0.09867
                                       -0.01950    0.00000    0.14403
multipole    44   42   46               0.14897
                                        0.07697    0.00000    0.46667
                                       -0.22387
                                        0.00000   -0.51710
                                       -0.34700    0.00000    0.74097
multipole    44   42   48               0.17897
                                        0.07697    0.00000    0.46667
                                       -0.22387
                                        0.00000   -0.51710
                                       -0.34700    0.00000    0.74097
multipole    45   44   42               0.03392
                                       -0.01463    0.00000   -0.07700
                                       -0.00706
                                        0.00000   -0.01579
                                       -0.02154    0.00000    0.02285
multipole    46   44   47              -0.17399
                                        0.00000    0.00000    0.32694
                                       -0.23348
                                        0.00000   -0.31115
                                       -0.04183    0.00000    0.54463
multipole    47   46   44               0.07185
                                        0.00000    0.00000   -0.07829
                                       -0.02252
                                        0.00000   -0.03344
                                        0.01606    0.00000    0.05596
multipole    48   50   44              -0.13175
                                        0.27622    0.00000    0.18608
                                        0.22471
                                        0.00000   -0.57970
                                       -0.16099    0.00000    0.35499
multipole    49   48   50               0.06360
                                        0.00000    0.00000   -0.07581
                                       -0.01386
                                        0.00000   -0.02607
                                        0.00078    0.00000    0.03993
multipole    50   48   51              -0.16203
                                        0.00000    0.00000    0.32333
                                       -0.30615
                                        0.00000   -0.27981
                                        0.01123    0.00000    0.58596
multipole    51   50   48               0.05938
                                        0.00000    0.00000   -0.09019
                                       -0.01198
                                        0.00000   -0.03174
                                        0.00468    0.00000    0.04372
multipole    52   54   53              -0.53467
                                       -0.11997    0.00000    0.30509
                                       -0.12904
                                        0.00000   -0.41083
                                       -0.24252    0.00000    0.53987
multipole    53   52   54               0.34832
                                       -0.02310    0.00000   -0.09243
                                        0.02057
                                        0.00000   -0.02589
                                        0.01814    0.00000    0.00532
multipole    54   52   55               0.18465
                                       -0.00129    0.00000    0.27865
                                       -0.41290
                                        0.00000   -0.18381
                                        0.28296    0.00000    0.59671
multipole    55   54   52              -0.00834
                                        0.00129    0.00000   -0.09710
                                       -0.00470
                                        0.00000   -0.00889
                                        0.01035    0.00000    0.01359
multipole    56   54   56              -0.12383
                                       -0.03880    0.00000    0.27670
                                       -0.21286
                                        0.00000   -0.18142
                                       -0.04969    0.00000    0.39428
multipole    57   56   54               0.04295
                                       -0.00542    0.00000   -0.10152
                                       -0.01487
                                        0.00000   -0.01382
                                       -0.00216    0.00000    0.02869
multipole    58   59   59              -0.32494
                                        0.53657    0.00000    0.37202
                                        0.44076
                                        0.00000   -0.87825
                                        0.00127    0.00000    0.43749
multipole    59   58   59               0.11615
                                       -0.19923    0.00000    0.19298
                                       -0.23681
                                        0.00000   -0.41373
                                       -0.10899    0.00000    0.65054
multipole    60   59   58               0.01544
                                        0.04231    0.00000   -0.02407
                                       -0.01587
                                        0.00000   -0.05904
                                        0.03668    0.00000    0.07491
#multipole   61   62  -62  -62          -0.57960
                                        0.15842    0.00000    0.06469
                                       -0.43288
                                        0.00000    0.27280
                                       -0.28801    0.00000    0.16008
#multipole   62   61  -62  -62           0.19320
                                       -0.03286    0.00000   -0.08851
                                       -0.04741
                                        0.00000    0.04478
                                        0.01653    0.00000    0.00263
#multipole    61  -62  -62  -62         -0.22011
                                        0.00000    0.00000    0.29395
                                        0.49102
                                        0.00000    0.49102
                                        0.00000    0.00000   -0.98204
#multipole    62   61  -62  -62          0.07337
                                       -0.01503    0.00000   -0.16505
                                       -0.05100
                                        0.00000    0.04892
                                        0.02991    0.00000    0.00208
multipole    61  -62  -62  -62         -0.22011
                                        0.00000    0.00000    0.29395
                                        0.35844
                                        0.00000    0.35844
                                        0.00000    0.00000   -0.71688
multipole    62   61  -62  -62          0.07337
                                       -0.01503    0.00000   -0.16505
                                       -0.05100
                                        0.00000    0.04892
                                        0.02991    0.00000    0.00208
multipole    63   64   64              -0.22892
                                        0.00000    0.00000    0.00000
                                        0.00000
                                        0.00000    0.00000
                                        0.00000    0.00000    0.00000
multipole    64   63   64               0.30723
                                        0.00000    0.00000    0.04225
                                       -0.07085
                                        0.00000   -0.07085
                                        0.00000    0.00000    0.14170
multipole    65   67  -66  -66         -0.73235
                                       -0.06766    0.00000    0.50053
                                       -0.78586
                                        0.00000    0.11843
                                       -0.42729    0.00000    0.66743
multipole    66   65   67               0.30411
                                       -0.01655    0.00000   -0.04238
                                        0.05116
                                        0.00000    0.01731
                                       -0.03326    0.00000   -0.06847
multipole    67   65   68               0.13715
                                        0.00000    0.00000    0.30940
                                       -0.22698
                                        0.00000   -0.17804
                                        0.14974    0.00000    0.40502
multipole    68   67   65              -0.00434
                                        0.00527    0.00000   -0.12660
                                        0.09141
                                        0.00000    0.00598
                                       -0.00549    0.00000   -0.09739
multipole    69   71  -70  -70         -0.78457
                                       -0.13919    0.00000    0.43672
                                       -0.60031
                                        0.00000   -0.00688
                                       -0.14251    0.00000    0.60719
multipole    70   69   71               0.33429
                                        0.01550    0.00000   -0.07147
                                        0.09104
                                        0.00000    0.00656
                                        0.06677    0.00000   -0.09760
multipole    71   69   73               0.07533
                                        0.02126    0.00000    0.27273
                                       -0.11070
                                        0.00000   -0.44964
                                       -0.22608    0.00000    0.56034
multipole    71   69   77               0.03849
                                        0.02126    0.00000    0.27273
                                       -0.11070
                                        0.00000   -0.44964
                                       -0.22608    0.00000    0.56034
multipole    72   71   69               0.03287
                                       -0.00328    0.00000   -0.11206
                                        0.05082
                                        0.00000    0.01536
                                       -0.10743    0.00000   -0.06618
multipole    73   71   74              -0.13077
                                        0.00000    0.00000    0.17419
                                       -0.25148
                                        0.00000   -0.02794
                                       -0.03628    0.00000    0.27942
multipole    74   73   71               0.03523
                                        0.01099    0.00000   -0.09907
                                        0.07159
                                        0.00000    0.01975
                                       -0.06636    0.00000   -0.09134
multipole    75   77   76              -0.16203
                                        0.00000    0.00000    0.32333
                                       -0.30615
                                        0.00000   -0.27981
                                        0.01123    0.00000    0.58596
multipole    76   75   77               0.05938
                                        0.00000    0.00000   -0.09019
                                       -0.01198
                                        0.00000   -0.03174
                                        0.00468    0.00000    0.04372
multipole    77   75   71              -0.13155
                                        0.27622    0.00000    0.18608
                                        0.22471
                                        0.00000   -0.57970
                                       -0.16099    0.00000    0.35499
multipole    78   77   75               0.06360
                                        0.00000    0.00000   -0.07581
                                       -0.01386
                                        0.00000   -0.02607
                                        0.00078    0.00000    0.03993
multipole    79   80  -81  -81         -0.50428
                                        0.44556    0.00000   -0.40528
                                       -0.46816
                                        0.00000    0.41972
                                       -0.53357    0.00000    0.04844
multipole    80   79   81               0.31436
                                        0.00000    0.00000   -0.01317
                                       -0.01675
                                        0.00000   -0.11749
                                        0.00000    0.00000    0.13424
multipole    81   79   81               0.03586
                                       -0.08099    0.00000    0.35884
                                       -0.28498
                                        0.00000   -0.43341
                                        0.18616    0.00000    0.71839
multipole    82   81   79               0.01970
                                       -0.00547    0.00000   -0.11748
                                        0.03191
                                        0.00000    0.06495
                                       -0.07663    0.00000   -0.09686
multipole    83   84  -84  -84         -0.19506
                                        0.64617    0.00000    0.22216
                                       -0.71281
                                        0.00000    0.37641
                                       -0.42297    0.00000    0.33640
multipole    84   83  -84  -84          0.00472
                                       -0.12628    0.00000    0.36014
                                       -0.65998
                                        0.00000   -0.16196
                                        0.18288    0.00000    0.82194
multipole    85   84   83               0.02010
                                        0.00237    0.00000   -0.10127
                                        0.04286
                                        0.00000    0.02869
                                       -0.01037    0.00000   -0.07155
multipole    86   87  -90  -90         -0.35208
                                        0.53462    0.00000   -0.10752
                                       -0.37199
                                        0.00000    0.11075
                                       -0.27388    0.00000    0.26124
multipole    87   86   90               0.10040
                                       -0.01272    0.00000   -0.10118
                                       -0.00613
                                        0.00000   -0.01791
                                       -0.00220    0.00000    0.02404
multipole    88   90   88              -0.19964
                                        0.22355    0.00000    0.20213
                                        0.29715
                                        0.00000   -0.50142
                                       -0.18234    0.00000    0.20427
multipole    89   88   90               0.07729
                                        0.00000    0.00000   -0.06322
                                       -0.00431
                                        0.00000   -0.00800
                                        0.01053    0.00000    0.01231
multipole    90   86   88               0.04684
                                        0.24638    0.00000    0.26382
                                        0.13003
                                        0.00000   -0.69427
                                       -0.14259    0.00000    0.56424
multipole    91   90   86               0.06203
                                       -0.02660    0.00000   -0.11082
                                       -0.02038
                                        0.00000    0.00233
                                       -0.01687    0.00000    0.01805
multipole    92   97  -93  -93         -0.16386
                                        0.86090    0.00000    0.17248
                                       -0.08613
                                        0.00000    0.06267
                                       -0.63594    0.00000    0.02346
multipole    93   92   95               0.05185
                                        0.13228    0.00000    0.30434
                                        0.06680
                                        0.00000   -0.61847
                                       -0.12599    0.00000    0.55167
multipole    94   93   94               0.00527
                                        0.00406    0.00000   -0.10764
                                       -0.01340
                                        0.00000   -0.01393
                                        0.00177    0.00000    0.02733
multipole    95   93   95              -0.09279
                                        0.22151    0.00000    0.21417
                                        0.13087
                                        0.00000   -0.35686
                                       -0.20050    0.00000    0.22599
multipole    96   95   96               0.04373
                                        0.01196    0.00000   -0.08273
                                       -0.01875
                                        0.00000   -0.01896
                                        0.00134    0.00000    0.03771
multipole    97   92   98              -0.02496
                                        0.04979    0.00000    0.29801
                                       -0.38717
                                        0.00000   -0.44363
                                        0.10820    0.00000    0.83080
multipole    98   97   92               0.02490
                                       -0.00991    0.00000   -0.11456
                                       -0.01289
                                        0.00000   -0.01285
                                       -0.00610    0.00000    0.02574
multipole    99  101  102               0.68400
                                        0.35855    0.00000    0.35024
                                       -0.16615
                                        0.00000    0.05785
                                       -0.18250    0.00000    0.10830
multipole   100   99  101              -0.00966
                                        0.04692    0.00000   -0.16916
                                        0.04863
                                        0.00000    0.00186
                                        0.02467    0.00000   -0.05049
multipole   101   99  102              -0.66032
                                       -0.11037    0.00000   -0.02173
                                       -0.47187
                                        0.00000    0.06623
                                       -0.17201    0.00000    0.40564
multipole   102   99  101              -0.28852
                                       -0.05862    0.00000   -0.09300
                                        0.58777
                                        0.00000   -1.01585
                                       -0.12557    0.00000    0.42808
multipole   103  102   99               0.13725
                                        0.00573    0.00000   -0.15670
                                       -0.00132
                                        0.00000   -0.06288
                                        0.01422    0.00000    0.06420
multipole   104  105  106               0.69704
                                        0.32439    0.00000    0.34585
                                        0.04129
                                        0.00000   -0.08955
                                       -0.08531    0.00000    0.04826
multipole   105  104  106              -0.66360
                                       -0.08498    0.00000    0.01193
                                       -0.55576
                                        0.00000    0.07729
                                       -0.11662    0.00000    0.47847
multipole   106  104  105              -0.34707
                                        0.00433    0.00000   -0.17817
                                        0.76240
                                        0.00000   -0.95193
                                       -0.04518    0.00000    0.18953
multipole   107  106  104               0.13083
                                       -0.00017    0.00000   -0.15549
                                        0.00490
                                        0.00000   -0.06975
                                       -0.00360    0.00000    0.06485
multipole   108  104  105              -0.19238
                                       -0.05630    0.00000    0.24000
                                       -0.15101
                                        0.00000   -0.08245
                                        0.04874    0.00000    0.23346
multipole   109  108  104               0.08145
                                        0.00000    0.00000   -0.09216
                                       -0.01356
                                        0.00000   -0.02534
                                        0.00133    0.00000    0.03890
multipole   110  104  105              -0.12704
                                        0.08569    0.00000   -0.08840
                                        0.31357
                                        0.00000    0.26378
                                        0.01921    0.00000   -0.57735
multipole   111  110  104               0.08145
                                        0.00000    0.00000   -0.09216
                                       -0.01356
                                        0.00000   -0.02534
                                        0.00133    0.00000    0.03890
multipole   112  110  113              -0.16203
                                        0.00000    0.00000    0.32333
                                       -0.30615
                                        0.00000   -0.27981
                                        0.01123    0.00000    0.58596
multipole   113  112  110               0.05938
                                        0.00000    0.00000   -0.09019
                                       -0.01198
                                        0.00000   -0.03174
                                        0.00468    0.00000    0.04372
multipole   114  116  117               0.65857
                                        0.41598    0.00000    0.37265
                                       -0.18173
                                        0.00000    0.06910
                                       -0.16554    0.00000    0.11263
multipole   115  114  116              -0.01588
                                        0.04813    0.00000   -0.16671
                                        0.05978
                                        0.00000    0.00638
                                        0.02414    0.00000   -0.06616
multipole   116  114  117              -0.67916
                                       -0.09581    0.00000    0.00698
                                       -0.39376
                                        0.00000   -0.01911
                                       -0.14889    0.00000    0.41287
multipole   117  114  116              -0.19685
                                        0.03810    0.00000   -0.16343
                                        0.75512
                                        0.00000   -1.07948
                                       -0.17807    0.00000    0.32436
multipole   118  117  114               0.06926
                                       -0.07490    0.00000   -0.17460
                                        0.02296
                                        0.00000   -0.06128
                                        0.00972    0.00000    0.03832
multipole   119  117  114              -0.00817
                                       -0.03257    0.00000    0.26271
                                       -0.36503
                                        0.00000   -0.20015
                                        0.00659    0.00000    0.56518
multipole   120  119  117               0.05741
                                        0.00000    0.00000   -0.08912
                                       -0.02815
                                        0.00000   -0.02091
                                       -0.00421    0.00000    0.04906
multipole   121  117  114               0.12999
                                        0.04290    0.00000    0.06605
                                        0.00941
                                        0.00000    0.03131
                                       -0.04157    0.00000   -0.04072
multipole   122  121  117               0.03407
                                       -0.00588    0.00831    0.04151
                                        0.04980
                                        0.05442    0.03448
                                       -0.02032    0.04044   -0.08428
multipole   124  125  126               0.64220
                                        0.22706    0.00000    0.31459
                                       -0.50118
                                        0.00000    0.23539
                                        0.02285    0.00000    0.26579
multipole   125  124  126              -0.66524
                                       -0.05964    0.00000   -0.00965
                                       -0.38021
                                        0.00000   -0.05967
                                       -0.12838    0.00000    0.43988
multipole   126  124  125              -0.18752
                                        0.20427    0.00000   -0.34027
                                        1.06373
                                        0.00000   -0.82316
                                        0.04814    0.00000   -0.24057
multipole   127  126  124               0.07726
                                       -0.01582    0.00000   -0.17196
                                        0.02821
                                        0.00000   -0.05415
                                        0.00463    0.00000    0.02594
multipole   128  126  124              -0.03623
                                        0.00000    0.00000    0.23240
                                       -0.39078
                                        0.00000   -0.13890
                                       -0.04627    0.00000    0.52968
multipole   129  128  126               0.06087
                                       -0.00898    0.00000   -0.08765
                                       -0.00712
                                        0.00000   -0.03030
                                       -0.02713    0.00000    0.03742
multipole   130  124  125              -0.24387
                                        0.00000    0.00000    0.07193
                                        0.28325
                                        0.00000    0.19307
                                        0.29847    0.00000   -0.47632
multipole   131  130  124               0.07693
                                        0.00000    0.00000   -0.08930
                                       -0.00778
                                        0.00000   -0.03132
                                        0.00708    0.00000    0.03910
multipole   132  134  135               0.61546
                                        0.34152    0.00000    0.30025
                                       -0.28519
                                        0.00000    0.03194
                                       -0.18758    0.00000    0.25325
multipole   133  132  134              -0.03132
                                        0.05682    0.00000   -0.15943
                                        0.04940
                                        0.00000    0.01244
                                        0.02463    0.00000   -0.06184
multipole   134  132  135              -0.67725
                                       -0.03839    0.00000    0.05139
                                       -0.35361
                                        0.00000   -0.08422
                                       -0.04480    0.00000    0.43783
multipole   135  132  134              -0.19791
                                        0.08128    0.00000   -0.16063
                                        0.77250
                                        0.00000   -0.95072
                                       -0.07925    0.00000    0.17822
multipole   136  135  132               0.00688
                                       -0.02610    0.00000    0.31443
                                       -0.28042
                                        0.00000   -0.19094
                                       -0.04366    0.00000    0.47136
multipole   137  136  135               0.04621
                                       -0.02617    0.00000   -0.09444
                                       -0.03433
                                        0.00000   -0.01491
                                       -0.01887    0.00000    0.04924
multipole   138  139  140               0.80568
                                        0.31418    0.00000    0.30936
                                        0.16264
                                        0.00000   -0.34513
                                       -0.12966    0.00000    0.18249
multipole   139  138  140              -0.76654
                                       -0.03160    0.00000   -0.15691
                                       -0.57523
                                        0.00000    0.31642
                                       -0.04180    0.00000    0.25881
multipole   140  138  139              -0.27849
                                        0.08827    0.07010   -0.39278
                                        0.75371
                                        0.00000   -0.88152
                                        0.04478    0.00000    0.12781
multipole   141  140  138              -0.01070
                                       -0.07637    0.00000    0.33533
                                       -0.45884
                                        0.00000   -0.31611
                                       -0.02300    0.00000    0.77495
multipole   142  141  140               0.04803
                                        0.00000    0.00000   -0.11050
                                        0.00177
                                        0.00000   -0.01118
                                       -0.00738    0.00000    0.00941
multipole   143  138  139              -0.27148
                                       -0.02865    0.00000    0.25409
                                       -0.19166
                                        0.00000   -0.03693
                                        0.08931    0.00000    0.22859
multipole   144  143  138               0.08135
                                        0.00000    0.00000   -0.08222
                                       -0.00361
                                        0.00000   -0.01208
                                        0.00930    0.00000    0.01569
multipole   145  147  146              -0.46919
                                        0.17702    0.00000    0.02746
                                        0.30493
                                        0.00001   -0.49979
                                       -0.33195    0.00000    0.19486
multipole   146  145  147               0.25776
                                       -0.04996    0.00000   -0.05293
                                       -0.10496
                                        0.00000   -0.10483
                                       -0.02100    0.00000    0.20979
multipole   147  148  145               0.75480
                                        0.47467    0.00000    0.23486
                                       -0.10721
                                        0.00000   -0.02052
                                       -0.49018    0.00000    0.12773
multipole   148  147  145              -0.60103
                                       -0.07976    0.00000   -0.03242
                                       -0.43271
                                        0.00000    0.10680
                                       -0.13721    0.00000    0.32591
multipole   149  147  148               0.05766
                                        0.01745    0.00000   -0.16520
                                        0.04785
                                        0.00000   -0.00670
                                        0.00836    0.00000   -0.04115
multipole   150  152  151              -0.51449
                                        0.17023    0.00000   -0.01931
                                        0.41976
                                        0.00000   -0.46569
                                       -0.31987    0.00000    0.04593
multipole   151  150  152               0.24812
                                       -0.06534    0.00000   -0.06940
                                       -0.09928
                                        0.00000   -0.08341
                                       -0.04437    0.00000    0.18269
multipole   152  153  150               0.79309
                                        0.44903    0.00000    0.22771
                                        0.16786
                                        0.00000   -0.19671
                                       -0.31142    0.00000    0.02885
multipole   153  152  150              -0.61662
                                       -0.08296    0.00000   -0.00320
                                       -0.50843
                                        0.00000    0.12249
                                       -0.13621    0.00000    0.38594
multipole   154  152  153              -0.18319
                                       -0.02790    0.00000    0.35976
                                       -0.10461
                                        0.00000   -0.11692
                                        0.04626    0.00000    0.22153
multipole   155  154  152               0.09103
                                       -0.01445    0.00000   -0.09273
                                       -0.01359
                                        0.00702   -0.02416
                                        0.00000    0.00000    0.03775
multipole   156  157  158               0.50491
                                        0.00000    0.00000    0.40735
                                       -0.55471
                                        0.00000    0.20992
                                        0.00000    0.00000    0.34479
multipole   157  156  158              -0.51381
                                        0.00000    0.00000    0.03694
                                       -0.46753
                                        0.00000    0.14362
                                        0.00000    0.00000    0.32391
multipole   158  156  157               0.00445
                                        0.00826    0.00000   -0.13618
                                        0.04855
                                        0.00000   -0.02218
                                       -0.01283    0.00000   -0.02637
multipole   159  160  161               0.54719
                                        0.03558    0.00000    0.38474
                                       -0.51526
                                        0.00000    0.15339
                                       -0.05746    0.00000    0.36187
multipole   160  159  161              -0.54061
                                       -0.05927    0.00000    0.03696
                                       -0.54551
                                        0.00000    0.14242
                                       -0.10982    0.00000    0.40309
multipole   161  159  160              -0.22361
                                       -0.01192    0.00000    0.24635
                                       -0.04230
                                        0.00000   -0.13918
                                        0.05210    0.00000    0.18148
multipole   162  161  159               0.07963
                                        0.00000    0.00000   -0.08270
                                       -0.01659
                                        0.00000   -0.02290
                                       -0.00527    0.00000    0.03949
multipole   163  159  160              -0.02186
                                        0.02389    0.00000   -0.16073
                                        0.04310
                                        0.00000   -0.01011
                                        0.01929    0.00000   -0.03299
multipole   164 -165 -165               0.23084
                                        0.00000    0.00000    0.49866
                                        1.54610
                                        0.00000   -2.21649
                                        0.00000    0.00000    0.67039
multipole   165  164  165              -0.11542
                                       -0.01880    0.00000   -0.26701
                                       -0.00334
                                        0.00000   -0.06496
                                       -0.01793    0.00000    0.06830
multipole   166  167  168               0.10186
                                        0.41482    0.00000    0.31682
                                        1.29701
                                        0.00000   -2.44834
                                       -0.60310    0.00000    1.15133
multipole   167  166  168              -0.11359
                                       -0.01493    0.00000   -0.23983
                                        0.01575
                                        0.00000   -0.08528
                                       -0.02660    0.00000    0.06953
multipole   168  166  169              -0.20403
                                        0.00000    0.00000    0.08076
                                       -0.06672
                                        0.00000   -0.20158
                                        0.00470    0.00000    0.26830
multipole   169  168  166               0.07192
                                        0.01185    0.00000   -0.10796
                                       -0.00170
                                        0.00000   -0.02645
                                       -0.00552    0.00000    0.02815
multipole   170 -171 -171              -0.18536
                                        0.00000    0.00000    0.69845
                                        1.74494
                                        0.00000   -2.90227
                                        0.00000    0.00000    1.15733
multipole   171  170  171               0.09754
                                       -0.06505    0.00000    0.62751
                                       -0.38369
                                        0.00000   -0.35868
                                        0.07336    0.00000    0.74237
multipole   172  171  170              -0.00162
                                       -0.08135    0.00000   -0.10752
                                       -0.02832
                                        0.00000    0.01779
                                       -0.08079    0.00000    0.01053
multipole   173  173  174              -0.15378
                                        0.47182    0.00000    0.35804
                                        1.33557
                                        0.00000   -2.74527
                                       -0.28170    0.00000    1.40970
multipole   174  173  173               0.13830
                                       -0.02096    0.00000    0.87668
                                       -0.42121
                                        0.00000   -0.34241
                                        0.03686    0.00000    0.76362
multipole   175  174  173               0.00516
                                       -0.13471    0.00000   -0.16285
                                        0.03143
                                        0.00000    0.04698
                                       -0.14145    0.00000   -0.07841
multipole   176  178  177              -0.35646
                                        0.01158    0.00000    0.38562
                                        0.89799
                                        0.00000   -2.59055
                                       -0.61964    0.00000    1.69256
multipole   177  176  178               0.20683
                                        0.03656    0.00000    0.23967
                                       -0.17006
                                        0.00000   -0.22209
                                        0.08133    0.00000    0.39215
multipole   178  176  180               0.04522
                                       -0.01580    0.00000    0.23138
                                       -0.86708
                                        0.00000   -0.05614
                                        0.02517    0.00000    0.92322
multipole   179  178  176              -0.00157
                                        0.02645    0.00000   -0.01401
                                       -0.10756
                                        0.00000    0.01205
                                        0.03041    0.00000    0.09551
multipole   180  178  176               0.06207
                                        0.00000    0.00000    0.28242
                                       -0.17130
                                        0.00005   -0.17329
                                       -0.03605    0.00005    0.34459
multipole   181  180  178               0.01516
                                       -0.01903    0.00000   -0.04771
                                       -0.06086
                                        0.00000   -0.04888
                                        0.00395    0.00000    0.10974
multipole   182  185  183              -0.51231
                                        0.34596    0.00000    0.23797
                                        1.44128
                                        0.00000   -2.68119
                                       -0.42800    0.00000    1.23991
multipole   183  182  185               0.19455
                                       -0.00724    0.00000    0.87776
                                       -0.49136
                                        0.00000   -0.45808
                                        0.13728    0.00000    0.94944
multipole   184  183  182              -0.00140
                                       -0.08817    0.00000   -0.11639
                                       -0.01709
                                        0.00000    0.02898
                                       -0.09187    0.00000   -0.01189
multipole   185  182  187               0.21136
                                        0.05894    0.00000    1.08244
                                       -0.83410
                                        0.00000   -0.39960
                                       -0.28758    0.00000    1.23370
multipole   186  185  182              -0.01700
                                       -0.16197    0.00000   -0.05262
                                       -0.07296
                                        0.00000    0.03911
                                       -0.14370    0.00000    0.03385
multipole   187  185  182               0.08583
                                       -0.00087    0.00000    0.36631
                                       -0.15926
                                        0.00000   -0.24306
                                       -0.00657    0.00000    0.40232
multipole   188  187  185               0.01959
                                       -0.01842    0.00000   -0.02389
                                       -0.05455
                                        0.00000   -0.05942
                                        0.02613    0.00000    0.11397
multipole   189  190 -191 -191          0.28761
                                        0.34263    0.00000    0.97464
                                       -0.69101
                                        0.00000    0.85372
                                       -0.23783    0.00000   -0.16271
multipole   190  189 -191 -191         -0.63423
                                       -0.03435    0.00000    0.26755
                                       -0.60034
                                        0.00000   -0.57370
                                        0.09678    0.00000    1.17404
multipole   191  189  190              -0.05355
                                       -0.08789    0.00000    0.54241
                                       -0.28219
                                        0.00000   -0.59075
                                       -0.20358    0.00000    0.87294
multipole   192  191  189               0.07562
                                       -0.01297    0.00000   -0.15656
                                        0.08665
                                        0.00000    0.07386
                                        0.01776    0.00000   -0.16051
multipole   193  195  194               0.68421
                                        0.00000    0.00000    0.04448
                                        0.24144
                                        0.00000    0.24144
                                        0.00000    0.00000   -0.48288
multipole   194  193  195              -0.59947
                                        0.02332    0.00000    0.46306
                                       -0.52310
                                        0.00000   -0.57116
                                       -0.00192    0.00000    1.09426
multipole   195  193  196              -0.12328
                                        0.00000    0.00000    0.20364
                                        0.03024
                                        0.00000    0.03024
                                        0.00000    0.00000   -0.06048
multipole   196  195  193               0.07916
                                        0.04006    0.00000    0.13850
                                       -0.05327
                                        0.00000   -0.09352
                                        0.00000    0.00000    0.14679
multipole   197  199  198               0.68359
                                        0.00000    0.00000    0.04182
                                        0.19805
                                        0.00000    0.19805
                                        0.00000    0.00000   -0.39610
multipole   198  197  199              -0.60332
                                        0.01993    0.00000    0.47399
                                       -0.51735
                                        0.00000   -0.59284
                                       -0.00125    0.00000    1.11019
multipole   199  197  201              -0.13070
                                        0.14439    0.00000    0.17369
                                        0.06183
                                        0.00000    0.00175
                                       -0.05422    0.00000   -0.06358
multipole   199  197  203              -0.13070
                                        0.14439    0.00000    0.17369
                                        0.06183
                                        0.00000    0.00175
                                       -0.05422    0.00000   -0.06358
multipole   200  199  197               0.09377
                                        0.02557    0.00000    0.17130
                                       -0.02736
                                        0.00000   -0.07299
                                       -0.03928    0.00000    0.10035
multipole   201  199  202              -0.13840
                                        0.00164    0.00000    0.15569
                                       -0.07355
                                        0.00000    0.00209
                                        0.08902    0.00000    0.07146
multipole   202  201  199               0.06931
                                        0.03490    0.00000    0.14233
                                       -0.05085
                                        0.00000   -0.10572
                                       -0.03686    0.00000    0.15657
multipole   203  205  199              -0.14395
                                        0.16557    0.00000    0.06166
                                       -0.06305
                                        0.00000   -0.14995
                                       -0.12323    0.00000    0.21300
multipole   204  203  205               0.10674
                                       -0.00429    0.00000    0.17162
                                       -0.00087
                                        0.00000   -0.10613
                                        0.05594    0.00000    0.10700
multipole   205  203  206              -0.14964
                                        0.00176    0.00000    0.05906
                                       -0.06180
                                        0.00000   -0.04998
                                        0.05205    0.00000    0.11178
multipole   206  205  203               0.04988
                                        0.04036    0.00000    0.15279
                                       -0.04562
                                        0.00000   -0.11014
                                       -0.01658    0.00000    0.15576
multipole   207  208    0               0.45374
                                        0.00000    0.00000    0.17403
                                        0.02679
                                        0.00000    0.02679
                                        0.00000    0.00000   -0.05358
multipole   208  207    0              -0.55384
                                        0.00000    0.00000   -0.27340
                                        0.23632
                                        0.00000    0.23632
                                        0.00000    0.00000   -0.47264
multipole   209  207    0               0.10010
                                        0.00000    0.00000   -0.24877
                                        0.13147
                                        0.00000    0.13147
                                        0.00000    0.00000   -0.26294
multipole   210  211    0               0.50893
                                        0.00000    0.00000    0.00838
                                        0.04512
                                        0.00000    0.04512
                                        0.00000    0.00000   -0.09024
multipole   211  210    0              -0.64934
                                        0.00000    0.00000   -0.41663
                                        0.40285
                                        0.00000    0.40285
                                        0.00000    0.00000   -0.80570
multipole   212  210    0              -0.16118
                                        0.00000    0.00000    0.17034
                                       -0.03526
                                        0.00000   -0.03526
                                        0.00000    0.00000    0.07052
multipole   213  212  210               0.10053
                                        0.00543    0.00000   -0.12249
                                        0.11906
                                        0.00000    0.00913
                                        0.02813    0.00000   -0.12819
multipole   214  216  214               0.04926
                                        0.00000    0.00000    0.09671
                                       -0.44194
                                        0.00000   -0.79642
                                        0.00000    0.00000    1.23836
multipole   215  214    0              -0.33505
                                        0.00000    0.00000    0.64421
                                       -0.23824
                                        0.00000   -0.23824
                                        0.00000    0.00000    0.47648
multipole   216  214  214              -0.14263
                                        0.00000    0.00000    0.00000
                                        0.79947
                                        0.00000   -1.59894
                                        0.00000    0.00000    0.79947
multipole   217 -217 -217              -0.08620
                                        0.00000    0.00000    0.13272
                                        0.52524
                                        0.00000   -1.18532
                                        0.00000    0.00000    0.66008
multipole   218  217  217               0.08620
                                        0.00000    0.00000    0.13804
                                       -0.01682
                                        0.00000   -0.15371
                                        0.00000    0.00000    0.17053
multipole   219  226  220               0.03072
                                        0.00125    0.00000    0.12469
                                       -0.06402
                                        0.00000   -0.27768
                                       -0.00103    0.00000    0.34170
multipole   220  219  221              -0.12168
                                        0.22018    0.00000    0.11136
                                        0.29801
                                        0.00000   -0.22028
                                        0.13973    0.00000   -0.07773
multipole   221  220  222              -0.13553
                                        0.08555    0.00000    0.06669
                                       -0.05093
                                        0.00000   -0.05737
                                       -0.14891    0.00000    0.10830
multipole   222  221  221              -0.13956
                                        0.09435    0.00000    0.05497
                                       -0.05052
                                        0.00000   -0.07647
                                       -0.15630    0.00000    0.12699
multipole   223  220  219               0.11676
                                       -0.00544    0.00000   -0.09789
                                       -0.00311
                                        0.00000   -0.00530
                                        0.00156    0.00000    0.00841
multipole   224  221  220               0.12064
                                        0.00158    0.00000   -0.09651
                                        0.00420
                                        0.00000   -0.00600
                                        0.00018    0.00000    0.00180
multipole   225  222  221               0.12262
                                        0.00000    0.00000   -0.09664
                                        0.00381
                                        0.00000   -0.00617
                                        0.00001    0.00000    0.00236
multipole   226  219  228              -0.33768
                                        0.28850    0.00000    0.23637
                                        0.34911
                                        0.00000   -0.68326
                                       -0.15432    0.00000    0.33415
multipole   227  226  219               0.17341
                                        0.00352    0.00000   -0.00147
                                       -0.01537
                                        0.00000   -0.02747
                                        0.00752    0.00000    0.04284
multipole   228  226  229              -0.48745
                                        0.00104    0.00000    0.34794
                                       -0.36178
                                        0.00000   -0.28743
                                        0.06023    0.00000    0.64921
multipole   229  228  226               0.16805
                                        0.01766    0.00000   -0.02781
                                       -0.01320
                                        0.00000   -0.02932
                                        0.01136    0.00000    0.04252
multipole   230  232  231              -0.60669
                                        0.25004    0.00000    0.31054
                                        0.14285
                                        0.00000   -0.56484
                                        0.08761    0.00000    0.42199
multipole   231  230  232               0.37742
                                       -0.15154    0.00000    0.12482
                                       -0.16870
                                        0.00000   -0.06439
                                       -0.05778    0.00000    0.23309
multipole   232  230  231               0.35048
                                       -0.18052    0.00000    0.28488
                                       -0.04061
                                        0.00000   -0.91018
                                       -0.10133    0.00000    0.95079
multipole   233  232  234              -0.16487
                                       -0.12931    0.00000   -0.04059
                                        0.25079
                                        0.00000   -0.74230
                                       -0.03793    0.00000    0.49151
multipole   234  233 -235               0.05310
                                        0.00000    0.00000    0.08000
                                        0.19556
                                        0.00000   -0.68026
                                       -0.32207    0.00000    0.48470
multipole   235 -234 -234              -0.08153
                                        0.00000    0.00000    0.00302
                                       -0.01236
                                        0.00000   -0.60883
                                       -0.03456    0.00000    0.62119
multipole   236  233  232               0.03926
                                        0.01720    0.00000   -0.00573
                                       -0.01641
                                        0.00000   -0.08767
                                        0.00931    0.00000    0.10408
multipole   237  234  233               0.03581
                                        0.04577    0.00000    0.00787
                                       -0.05397
                                        0.00000   -0.07608
                                        0.03840    0.00000    0.13005
multipole   238  235  234               0.03372
                                        0.00334    0.00000    0.01200
                                       -0.04259
                                        0.00000   -0.08852
                                       -0.00007    0.00000    0.13111
multipole   239  241  240              -0.15936
                                        0.00000    0.00000    0.17072
                                       -0.16041
                                        0.00000   -0.16041
                                        0.00000    0.00000    0.32082
multipole   240  239  241               0.07101
                                        0.02604    0.00000   -0.07852
                                        0.06399
                                        0.00000   -0.00135
                                        0.01071    0.00000   -0.06264
multipole   241 -242 -242              -0.04642
                                        0.00000    0.00000   -0.05822
                                        0.16134
                                        0.00000   -0.26055
                                       -0.02115    0.00000    0.09921
multipole   242 -241 -243               0.04308
                                       -0.03476    0.00000    0.15673
                                        0.00214
                                        0.00000   -0.48756
                                        0.06534    0.00000    0.48542
multipole   243 -244 -242              -0.03035
                                        0.02088    0.00000   -0.07104
                                        0.32973
                                        0.00000   -0.33538
                                       -0.03788    0.00000    0.00565
multipole   244 -243 -243              -0.03295
                                        0.00000    0.00000   -0.04604
                                        0.25750
                                        0.00000   -0.32503
                                        0.00000    0.00000    0.06753
multipole   245  242  241              -0.00539
                                        0.00087    0.00000   -0.13040
                                        0.06646
                                        0.00000   -0.01180
                                        0.00651    0.00000   -0.05466
multipole   246  243  242               0.00351
                                        0.00395    0.00000   -0.12499
                                        0.04823
                                        0.00000   -0.03089
                                        0.00000    0.00000   -0.01734
multipole   247  244  243               0.00400
                                        0.00000    0.00000   -0.11767
                                        0.05367
                                        0.00000   -0.04960
                                        0.00000    0.00000   -0.00407
multipole   248  255  249              -0.61500
                                        0.20835    0.00000    0.36077
                                        0.06711
                                        0.00000   -0.54881
                                        0.08087    0.00000    0.48170
multipole   249  248  255               0.37682
                                       -0.18948    0.00000    0.10490
                                       -0.18861
                                        0.00000   -0.03668
                                       -0.09567    0.00000    0.22529
multipole   250  252  251              -0.08979
                                       -0.02031    0.00000    0.18981
                                       -0.15326
                                        0.00000   -0.14641
                                       -0.03012    0.00000    0.29967
multipole   251  250  252               0.04735
                                        0.01648    0.00000   -0.02086
                                       -0.00436
                                        0.00000   -0.06126
                                        0.01451    0.00000    0.06562
multipole   252 -253 -253              -0.07125
                                        0.00000    0.00000   -0.09346
                                       -0.28590
                                        0.00000   -0.31283
                                       -0.15734    0.00000    0.59873
multipole   253 -252 -254               0.05669
                                        0.00000    0.00000    0.07425
                                        0.19135
                                        0.00000   -0.71537
                                        0.81888    0.00000    0.52402
multipole   254  255  253              -0.16275
                                       -0.13693    0.00000   -0.08748
                                        0.24073
                                        0.00000   -0.87144
                                       -0.09255    0.00000    0.63071
multipole   255  248  249               0.35041
                                       -0.17499    0.00000    0.28203
                                       -0.06360
                                        0.00000   -0.99813
                                        0.12021    0.00000    1.06173
multipole   256  253  252               0.02291
                                       -0.11444    0.00000    0.00444
                                       -0.04484
                                        0.00000   -0.07496
                                       -0.08873    0.00000    0.11980
multipole   257  254  253               0.03653
                                       -0.03274    0.00000    0.00320
                                       -0.02466
                                        0.00000   -0.09300
                                       -0.02348    0.00000    0.11766
multipole   258  260  265               0.20401
                                        0.55110    0.00000    0.16949
                                        1.40903
                                        0.00000   -1.21521
                                        0.78253    0.00000   -0.19382
multipole   259  258  260               0.04290
                                        0.00000    0.00000   -0.28505
                                        0.08538
                                        0.00000    0.01295
                                        0.00497    0.00000   -0.09833
multipole   260  258  262               0.35555
                                        0.52610    0.00000    0.13591
                                        0.06732
                                        0.00000   -0.52680
                                       -0.33777    0.00000    0.45948
multipole   261  260  262               0.08711
                                        0.01834    0.00000   -0.17930
                                        0.04096
                                        0.00000    0.00418
                                        0.00934    0.00000   -0.04514
multipole   262  260  263              -0.33192
                                        0.68769    0.00000    0.17557
                                        0.46425
                                        0.00000   -0.27707
                                       -0.13096    0.00000   -0.18718
multipole   263  262  265              -0.14311
                                       -0.17200   -0.00011   -0.15840
                                        0.35808
                                        0.00000   -0.08018
                                       -0.59259    0.00000   -0.27790
multipole   264  263  265               0.00836
                                       -0.01874    0.00000   -0.24808
                                        0.07031
                                        0.00000    0.03601
                                       -0.01152    0.00000   -0.10632
multipole   265  258  263              -0.33219
                                       -0.25202    0.00000    0.13585
                                       -0.39588
                                        0.00000   -0.33590
                                       -0.48999    0.00000    0.73178
multipole   266  265  263               0.10929
                                        0.01897    0.00000   -0.16645
                                        0.02818
                                        0.00000    0.00282
                                        0.01569    0.00000   -0.03100
multipole   267  279  269              -0.08880
                                        0.15672    0.00000    0.31130
                                        0.22550
                                        0.00000   -0.54417
                                       -0.04357    0.00000    0.31867
multipole   268  267  269               0.10822
                                        0.02459    0.00000   -0.07191
                                        0.00262
                                        0.00000   -0.01821
                                        0.00804    0.00000    0.01559
multipole   269  267  279              -0.19864
                                        0.06214    0.00000    0.22704
                                       -0.31471
                                        0.00000   -0.11680
                                       -0.05010    0.00000    0.43151
multipole   270  269  267               0.08043
                                        0.02130    0.00000   -0.09095
                                       -0.00575
                                        0.00000   -0.01678
                                        0.00762    0.00000    0.02253
multipole   271  273  279              -0.12987
                                        0.03834    0.00000   -0.06383
                                        0.44552
                                        0.00000   -0.65766
                                        0.40880    0.00000    0.21214
multipole   272  271  279               0.14705
                                        0.00425    0.00000   -0.23584
                                        0.02977
                                        0.00000    0.02206
                                        0.00518    0.00000   -0.05183
multipole   273  275  271               0.41790
                                        0.31085    0.00000    0.22290
                                        0.05430
                                        0.00000   -0.25793
                                       -0.23315    0.00000    0.20363
multipole   274  273  271               0.09985
                                        0.00180    0.00000   -0.21845
                                        0.03729
                                        0.00000    0.02534
                                        0.00058    0.00000   -0.06263
multipole   275  273  277              -0.11640
                                        0.03616    0.00000   -0.09843
                                        0.51526
                                        0.00000   -0.58535
                                        0.26527    0.00000    0.07009
multipole   276  275  277               0.14883
                                        0.00731    0.00000   -0.23413
                                        0.03803
                                        0.00000    0.02230
                                        0.00516    0.00000   -0.06033
multipole   277  275  279               0.05480
                                       -0.12804    0.00000    0.24168
                                       -0.44188
                                        0.00000   -0.20380
                                        0.01046    0.00000    0.64568
multipole   278  277  279               0.08285
                                        0.00846    0.00000   -0.21673
                                        0.03054
                                        0.00012    0.02442
                                        0.00333   -0.00018   -0.05496
multipole   279  277  271               0.12470
                                        0.11648    0.00000   -0.22490
                                        0.55064
                                        0.00000   -0.31799
                                       -0.10188    0.00000   -0.23265
multipole   280  282  289              -0.10971
                                        0.04983    0.00000    0.22870
                                        0.42557
                                        0.00000   -0.91481
                                        0.81241    0.00000    0.48924
multipole   281  280  289               0.08652
                                       -0.00346    0.00000   -0.22097
                                        0.02323
                                        0.00000    0.01497
                                        0.00096    0.00000   -0.03820
multipole   282  280  283               0.12507
                                        0.05779    0.00000    0.35513
                                       -0.13962
                                        0.00000   -0.46271
                                        0.15746    0.00000    0.60233
multipole   283  282  288              -0.14474
                                       -0.06204    0.00000   -0.03779
                                        0.15000
                                        0.00000   -0.23905
                                        0.18939    0.00000    0.08905
multipole   284  288  285              -0.10590
                                       -0.07239    0.00000   -0.15945
                                        0.16811
                                        0.00008    0.08601
                                       -0.02311    0.00000   -0.25412
multipole   285  284  286               0.06829
                                        0.28429    0.00000    0.11920
                                        0.55043
                                        0.00000   -0.51069
                                        0.42363    0.00000   -0.03974
multipole   286  285  287              -0.04665
                                       -0.02624    0.00000    0.01259
                                        0.05707
                                        0.00000   -0.18029
                                       -0.11281    0.00000    0.12322
multipole   287  289  286              -0.10330
                                       -0.08090    0.00000   -0.07863
                                        0.05342
                                        0.00000   -0.02406
                                       -0.05229    0.00000   -0.02936
multipole   288  283  289              -0.02981
                                       -0.12262    0.00000    0.04028
                                        0.18403
                                        0.00000   -0.14808
                                        0.17023    0.00000   -0.03595
multipole   289  280  288               0.22912
                                       -0.01112    0.00000   -0.00350
                                       -0.50740
                                        0.00007   -0.70671
                                       -0.09817    0.00004    1.21411
multipole   290  282  283               0.02522
                                        0.00961    0.00000   -0.19443
                                        0.02801
                                        0.00000    0.01404
                                        0.00733    0.00000   -0.04205
multipole   291  283  288               0.00472
                                        0.00794    0.00000   -0.20459
                                        0.02913
                                        0.00000    0.02224
                                        0.01030    0.00000   -0.05137
multipole   292  284  285              -0.00669
                                        0.00390    0.00000   -0.19154
                                        0.02589
                                        0.00000    0.01733
                                        0.00004    0.00000   -0.04322
multipole   293  285  286               0.00213
                                       -0.00328    0.00000   -0.18010
                                        0.02710
                                        0.00000    0.00837
                                       -0.00305    0.00000   -0.03547
multipole   294  286  287               0.00299
                                        0.00077    0.00001   -0.18002
                                        0.02705
                                        0.00000    0.00804
                                        0.00100    0.00000   -0.03509
multipole   295  287  286               0.00274
                                        0.01150    0.00000   -0.18274
                                        0.02088
                                        0.00000    0.01393
                                        0.00556    0.00000   -0.03481
multipole   296  298  304              -0.05207
                                        0.12708    0.00000    0.30461
                                        0.51487
                                        0.00000   -1.03191
                                        0.99253    0.00000    0.51704
multipole   297  296  298               0.08064
                                        0.00294    0.00000   -0.22349
                                        0.02393
                                        0.00000    0.01197
                                       -0.00105    0.00000   -0.03590
multipole   298  296  299               0.13685
                                        0.07298    0.00000    0.42844
                                       -0.00583
                                        0.00000   -0.53665
                                        0.44410    0.00000    0.54248
multipole   299  298  305              -0.15430
                                       -0.31678    0.00000   -0.17576
                                        0.03495
                                        0.00000   -0.16816
                                        0.46592    0.00000    0.13321
multipole   300  301  305              -0.17594
                                       -0.32728    0.00000    0.10264
                                       -0.33730
                                        0.00000    0.25185
                                        0.41510    0.00000    0.08545
multipole   301  300  302              -0.00504
                                        0.08002    0.00000    0.01285
                                        0.19466
                                        0.00000   -0.27554
                                        0.05597    0.00000    0.08088
multipole   302  301  303               0.04799
                                        0.15354    0.00000    0.12274
                                        0.39808
                                        0.00000   -0.40924
                                        0.27737    0.00000    0.01116
multipole   303  302  304              -0.09861
                                       -0.12330    0.00000   -0.02034
                                       -0.05105
                                        0.00000   -0.02710
                                        0.04040    0.00000    0.07815
multipole   304  296  303               0.27603
                                       -0.03708    0.00000    0.10586
                                       -0.40205
                                        0.00000   -0.75526
                                       -0.21214    0.00000    1.15731
multipole   305  299  300              -0.00524
                                        0.24654    0.00000   -0.13229
                                        0.14792
                                        0.00000   -0.04375
                                       -0.13945    0.00000   -0.10417
multipole   306  298  296               0.01133
                                       -0.01101    0.00000   -0.20742
                                        0.02681
                                        0.00000    0.01883
                                       -0.01044    0.00000   -0.04564
multipole   307  300  301              -0.01588
                                        0.00406    0.00000   -0.20022
                                        0.03446
                                        0.00000   -0.00468
                                        0.00261    0.00000   -0.02978
multipole   308  301  300              -0.00347
                                        0.00344    0.00000   -0.18481
                                        0.02834
                                        0.00000    0.00877
                                        0.00321    0.00000   -0.03711
multipole   309  302  301              -0.00474
                                       -0.00036    0.00000   -0.18495
                                        0.02870
                                        0.00000    0.00866
                                       -0.00087    0.00000   -0.03736
multipole   310  303  302              -0.00318
                                        0.00974    0.00000   -0.18655
                                        0.02342
                                        0.00000    0.01336
                                        0.00229    0.00000   -0.03678
multipole   311  313  299              -0.17670
                                       -0.03221    0.00000    0.13464
                                       -0.23859
                                        0.00000   -0.22495
                                        0.14777    0.00000    0.46354
multipole   312  311  299               0.06763
                                        0.01091    0.00000   -0.05795
                                       -0.00546
                                        0.00000   -0.01685
                                        0.00682    0.00000    0.02231
multipole   313  311  314              -0.17176
                                        0.01465    0.00000    0.14482
                                       -0.22206
                                        0.00000   -0.17168
                                        0.06871    0.00000    0.39374
multipole   314  313  311               0.05961
                                        0.01793    0.00000   -0.09695
                                        0.00132
                                        0.00000   -0.02366
                                        0.00756    0.00000    0.02234
multipole   315  317  323              -0.34013
                                        0.48521    0.00000    0.42107
                                        0.38065
                                        0.00000   -0.90200
                                        0.57132    0.00000    0.52135
multipole   316  315  317               0.36903
                                        0.00295    0.00000   -0.04986
                                       -0.00811
                                        0.00000   -0.00143
                                        0.00173    0.00000    0.00954
multipole   317  315  318               0.05501
                                        0.05932    0.00000    0.35325
                                       -0.29521
                                        0.00000   -0.38209
                                       -0.06382    0.00000    0.67730
multipole   318  317  324              -0.17824
                                        0.13669    0.00000   -0.00640
                                       -0.26881
                                        0.00000   -0.11674
                                        0.25457    0.00000    0.38555
multipole   319  320  324              -0.15513
                                       -0.10798    0.00000    0.03778
                                       -0.31271
                                        0.00000    0.18975
                                        0.01944    0.00000    0.12296
multipole   320  319  321              -0.04396
                                        0.11398    0.00000    0.07385
                                        0.07401
                                        0.00000   -0.22656
                                       -0.06641    0.00000    0.15255
multipole   321  320  322              -0.07791
                                        0.02176    0.00000    0.02355
                                       -0.03649
                                        0.00000   -0.07312
                                       -0.22343    0.00000    0.10961
multipole   322  321  323              -0.13725
                                       -0.03060    0.00000   -0.00278
                                       -0.04499
                                        0.00000   -0.01141
                                       -0.17084    0.00000    0.05640
multipole   323  315  322               0.29737
                                       -0.00949    0.00000   -0.31549
                                       -0.94644
                                        0.00000   -1.04582
                                        0.44720    0.00000    1.99226
multipole   324  318  319              -0.05438
                                        0.13187    0.00000   -0.03232
                                        0.30602
                                        0.00000   -0.00200
                                        0.05118    0.00000   -0.30402
multipole   325  317  315               0.08202
                                        0.00030    0.00000   -0.17548
                                        0.03614
                                        0.00000    0.03931
                                        0.00226    0.00000   -0.07545
multipole   326  319  324               0.07271
                                        0.00365    0.00000   -0.17332
                                        0.04281
                                        0.00000    0.03393
                                        0.00107    0.00000   -0.07674
multipole   327  320  321               0.04407
                                       -0.00064    0.00000   -0.16646
                                        0.04222
                                        0.00000    0.03302
                                       -0.00015    0.00000   -0.07524
multipole   328  321  322               0.04260
                                       -0.00047    0.00000   -0.16643
                                        0.04227
                                        0.00000    0.03310
                                       -0.00014    0.00000   -0.07537
multipole   329  322  323               0.04544
                                       -0.00629    0.00000   -0.16669
                                        0.03235
                                        0.00000    0.03842
                                       -0.00279    0.00000   -0.07077
multipole   330  331  318               0.68296
                                        0.09638    0.00038    0.34533
                                       -0.33906
                                       -0.00116   -0.23145
                                        0.17792    0.00032    0.57051
multipole   331  330  318              -0.64204
                                       -0.13596    0.00000   -0.03726
                                       -0.71286
                                        0.00000    0.41482
                                       -0.24172    0.00000    0.29804
multipole   332  330  331              -0.06217
                                        0.00976    0.00000   -0.15720
                                        0.04404
                                        0.00000    0.03288
                                        0.00320    0.00000   -0.07692
multipole   333  335  334              -0.20566
                                        0.00836    0.00000    0.11713
                                        0.44050
                                        0.00000   -1.36729
                                       -0.03427    0.00000    0.92679
multipole   334  333  335               0.25307
                                        0.02830    0.00000    0.06289
                                       -0.01360
                                        0.00000   -0.12859
                                        0.00625    0.00000    0.14219
multipole   335 -333 -333               0.15996
                                        0.00000    0.00000    0.02424
                                        0.39018
                                        0.00000   -0.79214
                                        0.00000    0.00000    0.40196
multipole   336  335  337              -0.08387
                                        0.00000    0.00000   -0.01676
                                        0.47376
                                        0.00000   -1.29130
                                        0.00000    0.00000    0.81754
multipole   337 -338 -336              -0.09524
                                        0.03212    0.00000    0.20067
                                        0.67177
                                        0.00000   -1.18961
                                        0.03229    0.00000    0.51784
multipole   338 -337 -339              -0.07039
                                        0.00000    0.00000    0.16226
                                        0.51509
                                        0.00000   -1.05821
                                        0.00000    0.00000    0.54312
multipole   339 -338 -338              -0.06499
                                        0.00000    0.00000    0.15549
                                        0.50678
                                        0.00000   -1.03720
                                        0.00000    0.00000    0.53042
multipole   340  337  338               0.14638
                                       -0.01944    0.00000    0.13578
                                        0.02066
                                        0.00000   -0.13729
                                        0.01944    0.00000    0.11663
multipole   341  338  337               0.14253
                                        0.00000    0.00000    0.13208
                                       -0.01770
                                        0.00000   -0.12726
                                        0.00000    0.00000    0.14496
multipole   342  339  338               0.14138
                                        0.00000    0.00000    0.13189
                                       -0.01988
                                        0.00000   -0.12630
                                        0.00000    0.00000    0.14618
multipole   343 -344 -344              -0.05502
                                        0.00000    0.00000    0.06242
                                        0.47168
                                        0.00000   -1.01871
                                        0.00000    0.00000    0.54703
multipole   344  343  345               0.05590
                                        0.08253    0.00000    0.13200
                                        0.46183
                                        0.00000   -0.87034
                                        0.01814    0.00000    0.40851
multipole   345  344  346              -0.01443
                                        0.07830    0.00000    0.05860
                                        0.48262
                                        0.00000   -0.93236
                                        0.02953    0.00000    0.44974
multipole   346 -345 -345               0.00018
                                        0.00000    0.00000    0.08868
                                        0.41966
                                        0.00000   -0.89464
                                        0.00000    0.00000    0.47498
multipole   347  343  344               0.25368
                                        0.00000    0.00000    0.04333
                                       -0.02775
                                        0.00000   -0.11523
                                        0.00000    0.00000    0.14298
multipole   348  344  343               0.14841
                                       -0.00208    0.00000    0.08930
                                       -0.03100
                                        0.00000   -0.13300
                                        0.00598    0.00000    0.16400
multipole   349  345  344               0.14173
                                       -0.00147    0.00000    0.09438
                                       -0.03561
                                        0.00000   -0.13066
                                        0.00307    0.00000    0.16627
multipole   350  346  345               0.13794
                                        0.00000    0.00000    0.09607
                                       -0.03544
                                        0.00000   -0.12801
                                        0.00000    0.00000    0.16345


      ########################################
      ##                                    ##
      ##  Dipole Polarizability Parameters  ##
      ##                                    ##
      ########################################


polarize      1          0.2050     0.3900
polarize      2          0.3950     0.3900
polarize      3          1.6400     0.3900
polarize      4          2.4800     0.3900
polarize      5          4.0400     0.3900
polarize      6          0.0280     0.3900
polarize      7          0.1200     0.3900
polarize      8          0.7800     0.3900
polarize      9          1.3500     0.3900
polarize     10          2.2600     0.3900
polarize     11          0.0800     0.0952
polarize     12          0.5500     0.1585
polarize     13          0.2600     0.2096
polarize     14          1.3500     0.3900
polarize     15          4.0000     0.3900
polarize     16          5.6500     0.3900
polarize     17          7.2500     0.3900
polarize     18          1.3440     0.3900     19
polarize     19          1.0730     0.3900     18
polarize     20          1.6000     0.3900     21
polarize     21          0.6000     0.3900     20
polarize     22          1.8280     0.3900     23
polarize     23          0.6000     0.3900     22
polarize     24          1.0730     0.3900     24
polarize     25          1.3340     0.3900     26
polarize     26          0.4960     0.3900     25
polarize     27          1.3340     0.3900     28
polarize     28          0.4960     0.3900     27
polarize     29          1.3340     0.3900     30
polarize     30          0.4960     0.3900     29
polarize     31          1.3340     0.3900     32
polarize     32          0.4960     0.3900     31
polarize     33          1.3340     0.3900     34
polarize     34          0.4960     0.3900     33
polarize     35          1.3340     0.3900
polarize     36          0.8370     0.3900     37
polarize     37          0.4960     0.3900     36
polarize     38          0.8370     0.3900     39   40
polarize     39          0.4960     0.3900     38
polarize     40          1.3340     0.3900     38   41
polarize     41          0.4960     0.3900     40
polarize     42          0.8340     0.3900     43   44   48
polarize     43          0.4960     0.3900     42
polarize     44          1.3340     0.3900     42   45
polarize     45          0.4960     0.3900     44
polarize     46          1.3340     0.3900     47
polarize     47          0.4960     0.3900     46
polarize     48          1.3340     0.3900     42   49
polarize     49          0.4960     0.3900     48
polarize     50          1.3340     0.3900     51
polarize     51          0.4960     0.3900     50
polarize     52          0.8340     0.3900     53   54
polarize     53          0.4960     0.3900     52
polarize     54          1.3340     0.3900     52   55
polarize     55          0.4960     0.3900     54
polarize     56          1.3340     0.3900     57
polarize     57          0.4960     0.3900     56
polarize     58          0.8370     0.3900     59
polarize     59          1.3340     0.3900     58   60
polarize     60          0.4960     0.3900     59
polarize     61          1.0730     0.3900     62
polarize     62          0.4960     0.3900     61
polarize     63          1.0730     0.3900     64
polarize     64          0.4960     0.3900     63
polarize     65          1.0730     0.3900     66   67
polarize     66          0.4960     0.3900     65
polarize     67          1.3340     0.3900     65   68
polarize     68          0.4960     0.3900     67
polarize     69          1.0730     0.3900     70   71
polarize     70          0.4960     0.3900     69
polarize     71          1.3340     0.3900     69   72
polarize     72          0.4960     0.3900     71
polarize     73          1.3340     0.3900     74
polarize     74          0.4960     0.3900     73
polarize     75          1.3340     0.3900     76
polarize     76          0.4960     0.3900     75
polarize     77          1.3340     0.3900     78
polarize     78          0.4960     0.3900     77
polarize     79          1.0730     0.3900     80   81
polarize     80          0.4960     0.3900     79
polarize     81          1.3340     0.3900     79   82
polarize     82          0.4960     0.3900     81
polarize     83          1.0730     0.3900     84
polarize     84          1.3340     0.3900     83   85
polarize     85          0.4960     0.3900     84
polarize     86          1.0730     0.3900     87
polarize     87          0.4960     0.3900     86
polarize     88          1.3340     0.3900     89
polarize     89          0.4960     0.3900     88
polarize     90          1.3340     0.3900     91
polarize     91          0.4960     0.3900     90
polarize     92          1.0730     0.3900     93   97
polarize     93          1.3340     0.3900     92   94
polarize     94          0.4960     0.3900     93
polarize     95          1.3340     0.3900     96
polarize     96          0.4960     0.3900     95
polarize     97          1.3340     0.3900     92   98
polarize     98          0.4960     0.3900     97
polarize     99          1.3340     0.3900    100  101  102
polarize    100          0.4960     0.3900     99
polarize    101          0.8370     0.3900     99
polarize    102          1.0730     0.3900     99  103
polarize    103          0.4960     0.3900    102
polarize    104          1.3340     0.3900    105  106
polarize    105          0.8370     0.3900    104
polarize    106          1.0730     0.3900    104  107
polarize    107          0.4960     0.3900    106
polarize    108          1.3340     0.3900    109
polarize    109          0.4960     0.3900    108
polarize    110          1.3340     0.3900    111
polarize    111          0.4960     0.3900    110
polarize    112          1.3340     0.3900    113
polarize    113          0.4960     0.3900    112
polarize    114          1.3340     0.3900    115  116  117
polarize    115          0.4960     0.3900    114
polarize    116          0.8370     0.3900    114
polarize    117          1.0730     0.3900    114  118
polarize    118          0.4960     0.3900    117
polarize    119          1.3340     0.3900    120
polarize    120          0.4960     0.3900    119  121
polarize    121          1.3340     0.3900    120
polarize    122          1.3340     0.3900    123
polarize    123          0.4960     0.3900    122
polarize    124          1.3340     0.3900    125  126
polarize    125          0.8370     0.3900    124
polarize    126          1.0730     0.3900    124  127
polarize    127          0.4960     0.3900    126
polarize    128          1.3340     0.3900    129
polarize    129          0.4960     0.3900    128
polarize    130          1.3340     0.3900    131
polarize    131          0.4960     0.3900    130
polarize    132          1.3340     0.3900    133  134  135
polarize    133          0.4960     0.3900    132
polarize    134          0.8370     0.3900    132
polarize    135          1.0730     0.3900    132
polarize    136          1.3340     0.3900    137
polarize    137          0.4960     0.3900    136
polarize    138          1.3340     0.3900    139  140
polarize    139          0.8370     0.3900    138
polarize    140          1.0730     0.3900    138
polarize    141          1.3340     0.3900    142
polarize    142          0.4960     0.3900    141
polarize    143          1.3340     0.3900    144
polarize    144          0.4960     0.3900    143
polarize    145          0.8370     0.3900    146  147
polarize    146          0.4960     0.3900    145
polarize    147          1.3340     0.3900    145  148  149
polarize    148          0.8370     0.3900    147
polarize    149          0.4960     0.3900    147
polarize    150          0.8370     0.3900    151  152
polarize    151          0.4960     0.3900    150
polarize    152          1.3340     0.3900    150  153
polarize    153          0.8370     0.3900    152
polarize    154          1.3340     0.3900    155
polarize    155          0.4960     0.3900    154
polarize    156          1.3340     0.3900    157  158
polarize    157          0.8370     0.3900    156
polarize    158          0.4960     0.3900    156
polarize    159          1.3340     0.3900    160  163
polarize    160          0.8370     0.3900    159
polarize    161          1.3340     0.3900    162
polarize    162          0.4960     0.3900    161
polarize    163          0.4960     0.3900    159
polarize    164          2.8000     0.3900    165
polarize    165          0.4960     0.3900    164
polarize    166          2.8000     0.3900    167  168
polarize    167          0.4960     0.3900    166
polarize    168          1.3340     0.3900    166  169
polarize    169          0.4960     0.3900    168
polarize    170          2.8000     0.3900    171
polarize    171          1.3340     0.3900    170  172
polarize    172          0.4960     0.3900    171
polarize    173          2.8000     0.3900    173  174
polarize    174          1.3340     0.3900    173  175
polarize    175          0.4960     0.3900    174
polarize    176          2.8000     0.3900    177  178
polarize    177          0.4960     0.3900    176
polarize    178          1.3340     0.3900    176  179
polarize    179          0.4960     0.3900    178
polarize    180          1.3340     0.3900    181
polarize    181          0.4960     0.3900    180
polarize    182          2.8000     0.3900    183  185
polarize    183          1.3340     0.3900    182  184
polarize    184          0.4960     0.3900    183
polarize    185          1.3340     0.3900    182  186
polarize    186          0.4960     0.3900    185
polarize    187          1.3340     0.3900    188
polarize    188          0.4960     0.3900    187
polarize    189          3.3000     0.3900    190  191
polarize    190          0.8370     0.3900    189
polarize    191          1.3340     0.3900    189  192
polarize    192          0.4960     0.3900    191
polarize    193          3.3000     0.3900    196
polarize    194          0.8370     0.3900    195
polarize    195          1.3340     0.3900    194
polarize    196          0.4960     0.3900    193
polarize    197          3.3000     0.3900    198
polarize    198          0.8370     0.3900    197
polarize    199          1.3340     0.3900    200
polarize    200          0.4960     0.3900    199
polarize    201          1.3340     0.3900    202
polarize    202          0.4960     0.3900    201
polarize    203          1.3340     0.3900    204
polarize    204          0.4960     0.3900    203
polarize    205          1.3340     0.3900    206
polarize    206          0.4960     0.3900    205
polarize    207          1.0730     0.3900    208
polarize    208          1.3340     0.3900    207  209
polarize    209          0.4960     0.3900    208
polarize    210          1.3340     0.3900    211  212
polarize    211          1.0730     0.3900    210
polarize    212          1.3340     0.3900    210  213
polarize    213          0.4960     0.3900    212
polarize    214          1.3340     0.3900    215  216
polarize    215          1.0730     0.3900    214
polarize    216          1.3340     0.3900    214
polarize    217          1.7500     0.3900    217  218
polarize    218          0.6960     0.3900    217
polarize    219          1.7500     0.3900    220
polarize    220          1.7500     0.3900    219  221  223
polarize    221          1.7500     0.3900    220  222  224
polarize    222          1.7500     0.3900    221  225
polarize    223          0.6960     0.3900    220
polarize    224          0.6960     0.3900    221
polarize    225          0.6960     0.3900    222
polarize    226          1.3340     0.3900    227
polarize    227          0.4960     0.3900    226
polarize    228          1.3340     0.3900    229
polarize    229          0.4960     0.3900    228
polarize    230          0.8370     0.3900    231  232
polarize    231          0.4960     0.3900    230
polarize    232          1.7500     0.3900    230  233
polarize    233          1.7500     0.3900    232  234  236
polarize    234          1.7500     0.3900    233  235  237
polarize    235          1.7500     0.3900    234  238
polarize    236          0.6960     0.3900    233
polarize    237          0.6960     0.3900    234
polarize    238          0.6960     0.3900    235
polarize    239          1.3340     0.3900    240
polarize    240          0.4960     0.3900    239
polarize    241          1.7500     0.3900    242
polarize    242          1.7500     0.3900    241  243  245
polarize    243          1.7500     0.3900    242  244  246
polarize    244          1.7500     0.3900    243  247
polarize    245          0.6960     0.3900    242
polarize    246          0.6960     0.3900    243
polarize    247          0.6960     0.3900    244
polarize    248          0.8730     0.3900    249  255
polarize    249          0.4960     0.3900    248
polarize    250          1.3340     0.3900    251  252
polarize    251          0.4960     0.3900    250
polarize    252          1.7500     0.3900    250  253
polarize    253          1.7500     0.3900    252  254  256
polarize    254          1.7500     0.3900    253  255  257
polarize    255          1.7500     0.3900    248  254
polarize    256          0.6960     0.3900    253
polarize    257          0.6960     0.3900    254
polarize    258          1.0730     0.3900    259  260  265
polarize    259          0.4960     0.3900    258
polarize    260          1.3340     0.3900    258  261  262
polarize    261          0.4960     0.3900    260
polarize    262          1.0730     0.3900    260  263
polarize    263          1.3340     0.3900    262  264  265
polarize    264          0.4960     0.3900    263
polarize    265          1.3340     0.3900    258  263  266
polarize    266          0.4960     0.3900    265
polarize    267          1.3770     0.3900    268
polarize    268          0.4960     0.3900    267
polarize    269          1.3370     0.3900    270
polarize    270          0.4960     0.3900    269
polarize    271          1.0730     0.3900    272  273  279
polarize    272          0.4960     0.3900    271
polarize    273          1.3340     0.3900    271  274  275
polarize    274          0.4960     0.3900    273
polarize    275          1.0730     0.3900    273  276  277
polarize    276          0.4960     0.3900    275
polarize    277          1.3340     0.3900    275  278  279
polarize    278          0.4960     0.3900    277
polarize    279          1.3340     0.3900    271  277
polarize    280          1.0730     0.3900    281  282  289
polarize    281          0.4960     0.3900    280
polarize    282          1.3340     0.3900    280  283  290
polarize    283          1.3340     0.3900    282  288  291
polarize    284          1.3340     0.3900    285  288  292
polarize    285          1.3340     0.3900    284  286  293
polarize    286          1.3340     0.3900    285  287  294
polarize    287          1.3340     0.3900    286  289  295
polarize    288          1.3340     0.3900    283  284  289
polarize    289          1.3340     0.3900    280  287  288
polarize    290          0.4960     0.3900    282
polarize    291          0.4960     0.3900    283
polarize    292          0.4960     0.3900    284
polarize    293          0.4960     0.3900    285
polarize    294          0.4960     0.3900    286
polarize    295          0.4960     0.3900    287
polarize    296          1.0730     0.3900    297  298  304
polarize    297          0.4960     0.3900    296
polarize    298          1.3340     0.3900    296  299  306
polarize    299          1.3340     0.3900    298  305
polarize    300          1.3340     0.3900    301  305  307
polarize    301          1.3340     0.3900    300  302  308
polarize    302          1.3340     0.3900    301  303  309
polarize    303          1.3340     0.3900    302  304  310
polarize    304          1.3340     0.3900    296  303  305
polarize    305          1.3340     0.3900    299  300  304
polarize    306          0.4960     0.3900    298
polarize    307          0.4960     0.3900    300
polarize    308          0.4960     0.3900    301
polarize    309          0.4960     0.3900    302
polarize    310          0.4960     0.3900    303
polarize    311          1.3340     0.3900    312
polarize    312          0.4960     0.3900    311
polarize    313          1.3340     0.3900    314
polarize    314          0.4960     0.3900    313
polarize    315          1.0730     0.3900    316  317  323
polarize    316          0.4960     0.3900    315
polarize    317          1.3340     0.3900    315  318  325
polarize    318          1.3340     0.3900    317  324  330
polarize    319          1.3340     0.3900    320  324  326
polarize    320          1.3340     0.3900    319  321  327
polarize    321          1.3340     0.3900    320  322  328
polarize    322          1.3340     0.3900    321  323  329
polarize    323          1.3340     0.3900    315  322  324
polarize    324          1.3340     0.3900    318  319  323
polarize    325          0.4960     0.3900    317
polarize    326          0.4960     0.3900    319
polarize    327          0.4960     0.3900    320
polarize    328          0.4960     0.3900    321
polarize    329          0.4960     0.3900    322
polarize    330          1.3340     0.3900    318  331  332
polarize    331          0.8370     0.3900    330
polarize    332          0.4960     0.3900    330
polarize    333          1.0730     0.3900    334  335
polarize    334          0.4960     0.3900    333
polarize    335          0.4960     0.3900    333  336
polarize    336          1.7500     0.3900    335  337
polarize    337          1.7500     0.3900    336  338  340
polarize    338          1.7500     0.3900    337  339  341
polarize    339          1.7500     0.3900    338  342
polarize    340          0.6960     0.3900    337
polarize    341          0.6960     0.3900    338
polarize    342          0.6960     0.3900    339
polarize    343          1.0730     0.3900    344  347
polarize    344          1.7500     0.3900    343  345  348
polarize    345          1.7500     0.3900    344  346  349
polarize    346          1.7500     0.3900    345  350
polarize    347          0.4960     0.3900    343
polarize    348          0.6960     0.3900    344
polarize    349          0.6960     0.3900    345
polarize    350          0.6960     0.3900    346

)**";
}
