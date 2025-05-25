import pstats
p = pstats.Stats('profile.out')
p.strip_dirs()
p.sort_stats('time')
# p.print_stats('./src')  # or part of your file path
p.print_stats(20)
