from pyspark.sql import functions as F
def normalize_text(c): return F.trim(F.regexp_replace(F.lower(F.coalesce(c,F.lit(""))),r"\s+"," "))
def category_expr(c):
 n=F.lower(F.coalesce(c,F.lit(""))); return F.when(n.rlike("camera|camcorder"),"camera").when(n.rlike("laptop|computer"),"computer").when(n.rlike("ssd|hard drive|memory card|gb|tb"),"storage_memory").when(n.rlike("speaker|soundbar|subwoofer"),"speaker").when(n.rlike("receiver|amplifier"),"receiver_amplifier").when(n.rlike("adapter|charger|battery|bag|keyboard|mouse"),"accessory").otherwise("other")
def accessory_expr(n,c): return F.when((c=="accessory")|F.lower(n).rlike("adapter|charger|battery|bag|keyboard|mouse"),1).otherwise(0)
def unix_or_timestamp(c): return F.coalesce(F.to_timestamp(F.from_unixtime(c.cast("long"))),F.to_timestamp(c.cast("string")))