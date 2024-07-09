from celery import Celery
from celery.schedules import crontab
import toml

config = toml.load("config.toml")
broker_url = config['celery']['broker_url']
schedule_cron = config['celery']['schedule_cron']

app = Celery('tasks', broker=broker_url, include=['tasks'])

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

minute, hour, day_of_month, month, day_of_week = schedule_cron.split()
app.conf.beat_schedule = {
    'clean-old-studies-every-day': {
        'task': 'tasks.run_clean_old_studies',
        'schedule': crontab(minute=minute, hour=hour, day_of_month=day_of_month, month_of_year=month, day_of_week=day_of_week),
    },
}

if __name__ == '__main__':
    app.start()
    