from functools import wraps
def traced(*features):
 def d(fn):
  @wraps(fn)
  def w(*a,**k):
   print("CALL",fn.__name__,"INPUTS",[getattr(x,"schema",type(x).__name__) for x in a],k); print("FEATURES",features); o=fn(*a,**k); print("OUTPUT",getattr(o,"schema",type(o).__name__)); return o
  return w
 return d