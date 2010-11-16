# Answer:
#
# Since the data file format is very regular ascii data, we may use the numpy function loadtxt with a custom defined data type
# 
#

student=np.dtype([('name',str,32),('gender',str,1),('class',int,1),('grade',int,(10))])

try:
    classdata=np.loadtxt('grades.txt',dtype=student)
except IOError:
    print "failed to load file"
    
print "%-10s %s %-8s" %("Name","Class","Average")
for i in classdata:
    print "%-10s %-5s % 3.2f" %(i['name'],i['class'],i['grades'].mean())

# class_average=classdata[:]['grades'].mean()

class_average=classdata['grades'].mean()

print "Class %s Male   Average % 3.2f" %(cn,male_average)
male_class_average=classdata[((classdata['gender']=='M'))]['grades'].mean()
print "Male     Average % 3.2f" %(cn,male_average)
female_class_average=classdata[((classdata['gender']=='F'))]['grades'].mean()
print "Female   Average % 3.2f" %(cn,male_average)

# You may also use the where and choose functions, though this is not as simple as building criterion
# male_average=np.choose(np.where(classdata[:]['gender']=='M'),classdata)['grades'].mean()
# female_average=np.choose(np.where(classdata[:]['gender']=='F'),classdata)['grades'].mean()

for cn in set(classdata['class']):
    male_average=classdata[((classdata['gender']=='M') & (classdata['class']==cn))]['grades'].mean()
    print "Class %s Male   Average % 3.2f" %(cn,male_average)
    female_average=classdata[((classdata['gender']=='F') & (classdata['class']==cn))]['grades'].mean()
    print "Class %s Female Average % 3.2f" %(cn,female_average)
