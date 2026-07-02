import numpy as np 
import cv2 
from .utils import draw_arc_from_three_points
import math

SCALE  = 20      # pixels per metre
H_MARGIN = 50
W_MARGIN =50                  # blank border around court (px)

# Standard ITF dimensions
COURT_W         = 50  # doubles width
COURT_H         = 94   # full length
PAINT_W         = 16 
PAINT_MARGIN    = 17 
PAINT_H         = 19 
FREE_T_R        = 5 
THREE_P_R       = 29
THREE_P_H       = 14 
THREE_P_W       = 44 
THREE_P_MARGIN  = 3 
CENTER_R_1      = 2
CENTER_R_2      = 6 


def _m2p(x_m, y_m,
         scale = SCALE, h_margin = H_MARGIN , w_margin = W_MARGIN) :
    """Convert metres → pixel (col, row)."""
    return (int(x_m * scale) + w_margin,
            int(y_m * scale) + h_margin)
def build_court_template(scale = SCALE,
                         h_margin = H_MARGIN ,HEAT = False, w_margin = W_MARGIN , BACKGROUND = (34, 187, 58)):


    W_px = int(COURT_W * scale) + 2 * w_margin
    H_px = int(COURT_H * scale) + 2 * h_margin

    img = np.zeros((H_px, W_px, 3), dtype=np.uint8)
    img[:] = BACKGROUND   # ITF green

    def mp(x, y):
        return _m2p(x, y, scale)

    WHITE = (255, 255, 255)
    T = 4
   # singles sideline offset

    # CORNERS 
    ctl = mp(0,           0)
    ctr = mp(COURT_W,     0)
    cbr = mp(COURT_W,     COURT_H)
    cbl = mp(0,           COURT_H)
    # THREE AREA POINTS TOP 
    TTLT = mp(THREE_P_MARGIN , 0 ) 
    TTRT = mp(COURT_W - THREE_P_MARGIN , 0 ) 
    TTLB = mp(THREE_P_MARGIN , THREE_P_H) 
    TTRB = mp(COURT_W - THREE_P_MARGIN ,THREE_P_H )
    TTR = mp(COURT_W / 2  , THREE_P_R)
    # THREE AREA POINTS BOTTOM 
    TBLB = mp(THREE_P_MARGIN , COURT_H ) 
    TBRB = mp(COURT_W - THREE_P_MARGIN , COURT_H ) 
    TBLT = mp(THREE_P_MARGIN , COURT_H -THREE_P_H )
    TBRT = mp(COURT_W - THREE_P_MARGIN , COURT_H -THREE_P_H )
    TBR = mp(COURT_W / 2  , COURT_H -  THREE_P_R) 

    # PAINT TOP 
    PTLT = mp(PAINT_MARGIN , 0 ) 
    PTRT = mp(COURT_W - PAINT_MARGIN , 0 )
    PTLB = mp(PAINT_MARGIN , PAINT_H ) 
    PTRB = mp(COURT_W - PAINT_MARGIN , PAINT_H )
    PTC = mp(COURT_W / 2  , PAINT_H)
    # PAINT BOTTOM  
    PBLT = mp(PAINT_MARGIN , COURT_H ) 
    PBRT = mp(COURT_W - PAINT_MARGIN , COURT_H )
    PBLB = mp(PAINT_MARGIN ,  COURT_H - PAINT_H ) 
    PBRB = mp(COURT_W - PAINT_MARGIN ,COURT_H - PAINT_H )
    PBC = mp(COURT_W / 2  , COURT_H -  PAINT_H) 

    # CENTER 
    CL = mp(0, COURT_H/2)
    CR = mp(COURT_W , COURT_H/2) 
    CC = mp(COURT_W /2 , COURT_H/2) 
    

    # Draw lines
    cv2.rectangle(img, ctl , cbr, WHITE, T)          # doubles outline
    cv2.rectangle(img, PTLT , PTRB, WHITE, T) 
    cv2.rectangle(img, PBLT , PBRB, WHITE, T)         # doubles outline
    cv2.line(img, TTLT,  TTLB,  WHITE, T)             # singles left
    cv2.line(img, TTRT, TTRB,  WHITE, T)  
    cv2.line(img, TBLT,  TBLB,  WHITE, T)             # singles left
    cv2.line(img, TBRT, TBRB,  WHITE, T) 
    cv2.ellipse(img, PTC, (int(FREE_T_R * scale) + w_margin  ,int(FREE_T_R * scale) + w_margin ) , 0, 0, 180, WHITE, T)
    cv2.ellipse(img, PBC, (int(FREE_T_R * scale) + w_margin  ,int(FREE_T_R * scale) + w_margin ) , 0, 0, -180, WHITE, T)
    img , top_arc_pts  = draw_arc_from_three_points(img , TTLB , TTRB , TTR , WHITE , T )
    img , bottom_arc_pts  = draw_arc_from_three_points(img , TBLT , TBRT ,TBR , WHITE , T )
    cv2.line(img, CL,  CR,  WHITE, T)
    cv2.circle(img , CC ,int(CENTER_R_1 * scale) + w_margin , WHITE , T ,1)
    cv2.circle(img , CC ,int(CENTER_R_2 * scale) + w_margin , WHITE , T ,1)
   

    points = {
        "ctl": ctl,
        "ctr": ctr,
        "cbr": cbr,
        "cbl": cbl,
        "TTLT": TTLT,
        "TTRT": TTRT,
        "TTLB": TTLB,
        "TTRB": TTRB,
        "TTR": TTR,
        "TBLB" :TBLB, 
        "TBRB": TBRB,
        "TBLT": TBLT,
        "TBRT": TBRT,
        "TBR": TBR,
        "PTLT": PTLT,
        "PTRT": PTRT,
        "PTLB": PTLB,
        "PTRB": PTRB,
        "PTC": PTC,
        "PBLT": PBLT,
        "PBRT": PBRT,
        "PBLB": PBLB,
        "PBRB": PBRB,
        "PBC": PBC,
        "CL": CL,
        "CR": CR,
        "CC": CC,
    }

    return img , points  , top_arc_pts , bottom_arc_pts

