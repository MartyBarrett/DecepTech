import pympi


#List all tiers in .eaf file
eaf = pympi.Elan.Eaf('annotations/1.BoL.An.Tr.eaf')
print("Available tiers:")
for tier in eaf.get_tier_names():
    print(f"- {tier}")
