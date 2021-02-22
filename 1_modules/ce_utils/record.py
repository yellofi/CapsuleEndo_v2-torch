#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import time

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def sec_to_m_s_ms(sec):
    """
    sec: time.time() - start_time
    output: ex) 00:47.421
    """
    min_sec = time.strftime("%M:%S", time.gmtime(sec))
    ms = '{:03d}'.format(int((sec - int(sec))*1000))   
    return '.'.join([min_sec, ms])

def values_from_str_time(str_time):
    """
    str_time: ex) 00:47.421
    """
    str_time.split('.')[0].split(':')

    sec = 60*int(str_time.split('.')[0].split(':')[0]) + int(str_time.split('.')[0].split(':')[1])
    msec = int(str_time.split('.')[1])

    return sec, msec

def total_sum_time(times):
    sec = 0
    msec = 0
    for time in times:
        sec1, msec1 = values_from_str_time(time)
        sec += sec1
        msec += msec1
        
    m, s = sec//60, sec%60
    s_, ms = msec//1000, msec%1000

    if (s+s_) >= 60:
        m_, s = (s+s_)//60, (s+s_)%60 
        m = m + m_
        
    elif (s+s_)<60:
        s = s + s_
    
    total_time = '{:02d}:{:02d}.{:03d}'.format(m, s, ms)
    return total_time