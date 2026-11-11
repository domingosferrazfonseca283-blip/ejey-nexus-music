# EJey Nexus Music

Estúdio móvel de criação musical assistida por IA.

## Estado atual

Primeira base móvel: interface responsiva, criação local de prompts, biblioteca persistente no dispositivo e preparação do endereço do motor de geração. A geração de áudio ainda não está ligada a um modelo.

## Executar no telemóvel

Abrir a aplicação através de um servidor HTTP local ou alojar a pasta como site estático e instalar pelo navegador como aplicação. Para uso com Termux, instalar Node.js no Termux e executar:

```sh
pkg update && pkg install nodejs
npx serve .
```

Abrir no navegador do telemóvel o endereço local indicado pelo servidor.

## APK Android

A embalagem APK será feita com Capacitor/Gradle num computador ou num ambiente CI. Termux é útil para executar ferramentas e testar o serviço local, mas não é o próprio APK.

## Arquitetura planeada

- Aplicação Android: interface, biblioteca local e gestão de tarefas.
- Lógica no telemóvel: composição/validação de prompts e fila de pedidos.
- Motor no PC: inferência de geração musical, com API local na rede.
- Treino futuro: adaptar um modelo aberto com dados licenciados ou criados com autorização. O treino de um modelo musical completo não é realista num telemóvel; o telefone será o controlador e o PC fará o treino pesado.

## Privacidade e direitos

Usar apenas letras, gravações, vozes e conjuntos de dados próprios ou com autorização. Não clonar a voz de alguém sem consentimento.

## Próximas etapas

1. Confirmar esta base móvel em Android.
2. Adicionar API do motor de áudio e reprodução/gestão de resultados.
3. Preparar o projeto do PC e um ZIP com scripts de instalação e inferência.
4. Avaliar fine-tuning de um modelo aberto, respeitando as licenças dos pesos e dos dados.
