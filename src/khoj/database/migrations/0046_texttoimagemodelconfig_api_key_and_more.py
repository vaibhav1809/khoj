# Generated by Django 4.2.11 on 2024-06-20 19:02

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("database", "0045_fileobject"),
    ]

    operations = [
        migrations.AddField(
            model_name="texttoimagemodelconfig",
            name="api_key",
            field=models.CharField(blank=True, default=None, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name="texttoimagemodelconfig",
            name="model_type",
            field=models.CharField(
                choices=[("openai", "Openai"), ("stability-ai", "Stabilityai")], default="openai", max_length=200
            ),
        ),
    ]