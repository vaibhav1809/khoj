# Generated by Django 4.2.11 on 2024-06-20 19:48

import django.db.models.deletion
from django.conf import settings
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
        migrations.CreateModel(
            name="UserPaintModelConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "setting",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="database.texttoimagemodelconfig"
                    ),
                ),
                (
                    "user",
                    models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
                ),
            ],
            options={
                "abstract": False,
            },
        ),
    ]
