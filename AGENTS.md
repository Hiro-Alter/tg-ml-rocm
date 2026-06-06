## Optimización de tokens y alcance de trabajo

Para reducir consumo de tokens y evitar trabajo innecesario:

* Trabaja por fases pequeñas.
* Antes de modificar archivos, presenta un plan breve con máximo 8 puntos.
* No analices todo el repositorio si no es necesario; inspecciona únicamente los archivos relevantes para la tarea actual.
* No reescribas archivos completos cuando baste con cambios puntuales.
* Evita explicaciones largas; resume los cambios en máximo 10 líneas.
* No generes documentación extensa salvo que se solicite explícitamente.
* No ejecutes entrenamientos largos, búsquedas completas de hiperparámetros ni comandos costosos sin confirmación.
* No uses subagentes ni paralelización automática salvo autorización.
* No crees funciones o módulos adicionales fuera del alcance solicitado.
* Si existe ambigüedad técnica, pregunta antes de implementar.
* Después de cada fase, detente y espera confirmación antes de continuar con la siguiente.
* Prioriza código funcional, mínimo y mantenible sobre una arquitectura demasiado compleja.

## Memoria del proyecto y continuidad entre sesiones

Para facilitar la continuidad si se agotan los tokens o se reinicia la sesión, mantener actualizados los siguientes archivos:

* `PROJECT_STATE.md`: resumen breve del estado actual del proyecto, decisiones técnicas tomadas, estructura implementada, comandos funcionales y problemas conocidos.
* `TODO.md`: lista de tareas pendientes organizada por fases.
* `EXPERIMENTS.md`: registro de experimentos de entrenamiento, configuración usada, métricas obtenidas y observaciones.
* `CHANGELOG.md`: cambios importantes realizados en el repositorio.

Reglas:

* Actualizar `PROJECT_STATE.md` al finalizar cada fase.
* Actualizar `TODO.md` marcando tareas completadas y dejando claras las siguientes.
* Actualizar `EXPERIMENTS.md` solo cuando haya resultados reales de entrenamiento o evaluación.
* No inventar métricas ni resultados.
* Mantener estos archivos breves, concretos y útiles para retomar el trabajo en otra sesión.
* No crear subagentes ni flujos paralelos salvo autorización explícita.
* Al iniciar una nueva sesión, leer primero `AGENTS.md`, `PROJECT_STATE.md` y `TODO.md` antes de modificar código.
