import sys
import os
import time
import math
import numpy as np

def main_tl_status_correct(stateCur, stateNext, stateCurDuration):
    stateOFF = 16
    stateUnknown = 15
    min_duration_state = 240 #2 --> Sec (fps * # sec)
    _statePred = stateCur

    if stateCur != stateNext:
        if stateCurDuration >= min_duration_state and (stateNext == stateUnknown or stateNext == stateOFF):
            _statePred = stateCur
        else:
            _statePred = stateNext
    else:
        _statePred = stateCur
    return _statePred

def sub_tl_status_correct(stateCur, stateNext, stateCurDuration, subStatus, blink):
    max_duration_state = 80 #2 --> Sec (fps * # sec)
    min_duration_state = 30
    _statePred = stateCur
    _blink = blink

    if stateCur != stateNext:
        if stateCurDuration <= max_duration_state and stateCurDuration >= min_duration_state:
            if stateNext in subStatus:
                _statePred = stateNext
                _blink = 1
            else:
                _statePred = stateCur
        else:
            _statePred = stateNext ## -> Yellow to Main
    else:
        _statePred = stateCur
    return _statePred, _blink

def tl_state_correction(trk, stateCur, stateNext, blink):
    _blink = blink
    _statePred = 0
    stateCurDuration = trk.hit_streak
    
    mainStatus = [0,1,2,3,5,6,7,16]
    subStatus = [4,8,9,10,11,12,13,14,15]

    #print(trk.hit_streak)

    if stateCur in mainStatus:
        _statePred = main_tl_status_correct(stateCur, stateNext, stateCurDuration)
    elif stateCur in subStatus:
        _statePred, _blink = sub_tl_status_correct(stateCur, stateNext, stateCurDuration, subStatus, _blink)
    _statePred = stateNext
    return _statePred, _blink 

