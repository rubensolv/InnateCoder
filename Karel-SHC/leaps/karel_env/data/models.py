import sys

try:
    from django.db import models
except Exception:
    print('Exception: Django Not Found, please install it with "pip install django".')
    sys.exit()

class Solutions(models.Model):
    search_space = models.CharField(max_length=50)
    search_method = models.CharField(max_length=70)
    num_iterations= models.IntegerField()
    task = models.CharField(max_length=70)
    solution = models.CharField(max_length=900)
    best_reward = models.FloatField()
    current_iteration = models.IntegerField(default=0)
    uuid = models.CharField(max_length=150, default='', null=True)
    crashable = models.BooleanField(null=True)
    leaps_behaviour = models.BooleanField(null=True)
    seed= models.IntegerField(null=True)
    
    def __str__(self):
        return self.search_method+str(self.num_iterations)+self.task


class Solutions_Leaps(models.Model):
    search_space = models.CharField(max_length=50)
    search_method = models.CharField(max_length=70)
    num_iterations= models.IntegerField()
    task = models.CharField(max_length=70)
    solution = models.CharField(max_length=900)
    best_reward = models.FloatField()
    current_iteration = models.IntegerField(default=0)
    id_task = models.CharField(max_length=200,null=True)
    state_task = models.CharField(max_length=2000,null=True)
    seed= models.IntegerField(null=True)
